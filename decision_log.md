# Decision log

The log was written as the work progressed and tidied at the end. It is also saved as `decision_log.md`.

**1. Alternatives rejected**

| Decision | Options considered | Chosen | The one reason that decided it |
|---|---|---|---|
| Aligning timestamps that differ by seconds | (a) cluster timestamps closer than 30 s into one poll; (b) exact-second join; (c) floor to the 1-minute slot | (c) | Sensor groups are polled at :13, :21–33, :51 and :53 each minute, so (a) chains them into one endless poll and (b) never joins anything. Collisions within a slot (mostly the twice-a-minute uptime counter) are counted, not hidden. |
| EDFA sampling interval | 1 min, 5 min, 15 min, 1 h | 5 min | Each optical sensor is missing from 20–30% of minutes, so 1 min is mostly gaps. 15 min or more would smear a minute-scale fibre event; 5 min gives 3–4 readings per median. |
| SFP reporting window | Current; Previous 15min; Current Day / Previous Day; a consensus of all four | Current, with the day windows as a cross-check | It is the only window with no impossible values that is aligned to poll time. Previous 15min's clock is 49–94 min fast and Current Day holds 24% impossible voltages, so neither can anchor a 15-min model. A consensus would hide the single-window steps that turned out to be the main SFP finding. |
| Outage handling | drop the rows; impute through them; mask with a guard | Mask optical values while ICMP loss = 100%, plus a 5-min guard, and never fill | Imputation invents data for an unreachable device, and dropping rows loses the timeline needed to tell outages from polling misses. |
| Detection method | Isolation Forest; autoencoder or LSTM; raw control charts; physics-informed residuals | Residuals of 1–3-coefficient models, with Isolation Forest only as a cross-check | Evaluated, not assumed: Isolation Forest caught none of the injected faults, alerted on 8/12 explained hot afternoons, and is blind to signals that were constant in training. |
| Threshold | fixed 3σ; a quantile of training scores; an alert budget | max(4σ, 99.9th percentile of the training persistent score), physical floors per signal, 15 min of persistence, 6 h incident merge | Quantised sensors have a near-zero spread, so without floors a single rounding step would alert. The floors encode "the smallest change an engineer cares about". |
| Unmapped SFP ids (41% of SFP rows) | drop; keep as unknown; infer and model | Keep, label, infer the role from values, do not model | The value signatures (−48 V-like feeds, a temperature correlating with the lasers at r ≈ 0.8–0.9) support an inference, but an unverified identity should not drive alerts. |

**2. Assumptions**

| Assumption | Status |
|---|---|
| Timestamps are UTC, as the brief says. Consistent with the SFP day windows closing at 18:30 UTC, which is midnight IST. | verified |
| Polling is nominally every 60 s (median inter-arrival 60 s). | verified |
| Both EDFAs are single-pump (Pump2 Itec is 0 at every reading). | verified |
| ICMP loss = 100% means unreachable, and optical values then are stale. 33% or 67% loss is reachable. | plausible |
| Unmapped SFP ids are shelf-level sensors (a chassis temperature, two −48 V feeds, their thresholds, status flags). | plausible |
| SFP Transmit Power is in mW (0.7–1.1 mW fits STM-4 optics at −1.5 to +0.4 dBm). | plausible |
| 18–21 Feb is normal for training: no outage, after the thermal event, before every level shift seen in Part C. | plausible |
| The 16–18 Feb heat was site-level (both units, r = 0.995, pump temperature held). | plausible |
| Steps that start at 00:00 UTC with off-grid values come from the monitoring path or are injected, not from the device. | plausible |
| The EDFA room's inverted daily cycle comes from scheduled HVAC. | guess |
| 70 °C is the SFP case limit (part numbers unknown). | guess |

**3. What surprised me**

* **Coverage.** Only 22 of 88 days carry EDFA optics, with two multi-week holes across all assets. I rebuilt the plan around two short blocks and a 3.5-day training window.
* **Unmapped ids.** The 60 unmapped SFP ids are not the two missing STM4 ports another asset's mapping suggested. Their values look like shelf power and temperature, so they were kept and inferred rather than dropped.
* **Window clock and corruption.** The SFP windows disagree, the 15-minute window's clock runs over an hour fast, and `Current Day` is corrupted. This turned "pick a window" into a consistency check that became a detector.
* **Step timing.** Several step changes start at exactly 00:00 UTC with values off the sensor's own resolution. That contradicts "values untouched" in the brief, so each one was judged on physics before being called a fault.
* **Cooler linearity.** Itec follows case temperature at r = 0.997. That is cleaner than a real thermo-electric cooler usually is, so I used it but noted it.
* **Naming and files.** `Output Power Threshold (PIN)` is really the POUT threshold. The delivered files are `*_anomalies_f.csv`, not `EDFA_data.csv` / `SFP_data.csv`. One unmapped id arrives 5 times per poll (130k exact duplicates). SFP ICMP loss contains values from −309 to +366.

**4. Where the time went**

*[Candidate: replace with your actual hours.]*

| Part | Hours |
|---|---|
| A. wrangling and inventory | _ |
| B. preprocessing | _ |
| C. insights | _ |
| D. detection and evaluation | _ |
| E. write-up | _ |

Part B and the SFP window question took the most time, because the windows had to be understood before they could be chosen.

**5. With ten times the data and two more weeks, in order of expected impact**

1. **A full year per unit, without collector gaps.** This would let the cooler model see the whole seasonal range, show real ageing slopes with honest lead times, and measure the false-alarm rate per season.
2. **Vendor limits.** Pump end-of-life current, TEC maximum current, and the SFP part numbers with their DDM alarm thresholds. These turn z-scores into a "margin to failure" that an engineer can plan around.
3. **Maintenance tickets as weak labels.** Use them to calibrate the floors and thresholds, and to check whether flagged shifts were devices or monitoring.
4. **Pooling across many units of the same class.** A hierarchical version of the same residual models, so a new unit inherits a prior instead of needing 3.5 clean days.
5. **A data-quality monitor upstream of the detector.** Window consistency, quantisation breaks and collector gaps, reported separately from device health.
6. Only then, a learned multivariate model, if the residuals show structure that the physics models miss.

**6. Deployment sketch**

* **How it runs.** A Python job, using pandas and numpy only, runs every 5 minutes on the site's NMS server. It reads the last 24 hours from the poller's database, applies the validity rules, resamples, and scores against per-asset coefficients stored as a small JSON file. It then writes incidents to the NMS alarm table, labelled "device" or "monitoring".
* **Retraining.** Coefficients are refitted monthly on the last 14 clean days (no outage, no open incident). A refit is also triggered immediately by a hardware swap, a firmware change or a set-point change, which the config and identity sensors reveal.
* **False alarms.** An engineer closes each incident with a reason code: real, explained or false. False windows are excluded from the next refit, and counts per model are reviewed monthly to adjust the floors.
* **Compute.** 1 vCPU, 1 GB of RAM and about 100 MB of disk per 100 assets. No GPU, no internet. The model is the smallest part of this system; the collector, the data-quality checks and the feedback loop matter more.
