# Insights: EDFA & SFP telemetry

Every number below is printed by a cell in this part.

**1. Only 22 of the 88 days in the files contain EDFA optical telemetry. The gaps are the monitoring system's, not the devices'.**

* From 2 to 20 Jan both EDFAs were unreachable 93.5% of the time (ICMP loss 100%) and reported no optical sensors at all. Optical telemetry first appears on 16 Feb at 10:41 UTC.
* On 46 days there are **no rows from any sensor on any of the three assets**: 21 Jan–12 Feb (23 days), 14–15 Feb (2 days) and 2–22 Mar (21 days).
* All assets drop out on the same days, so this is a collector or export gap, not three devices failing at once.
* The uptime counters never reset (both EDFAs were up 354 → 397 days), which confirms the devices kept running through every gap.

*Why it matters:* any baseline or trend for these EDFAs rests on two blocks of 13 and 9 days. A 3-week hole in the history of a live amplifier is itself a maintenance-visibility problem worth raising with the NMS team.

**2. Both EDFAs ran 7–9 °C hot for the first 44 hours of monitoring (16 Feb 10:41 to 18 Feb ~06:30 UTC). It was the site, not the amplifiers.**

* Case temperature medians were 37.7 °C on 010147e2 and 40.4 °C on 7a1d09b1, against 30.8 and 31.6 °C afterwards.
* The cooler (TEC) current doubled, to 349 and 386 mA from 173 and 161 mA.
* Both units moved together: their case temperatures correlate at r = 0.995. The pump chips stayed at 25.0–25.1 °C throughout, so the coolers had headroom and held regulation.

*Why it matters:* this was a site cooling event, probably HVAC, coinciding with the restoration of monitoring. A naive Itec threshold would have paged twice, once per unit, for a problem that belongs to facilities.

**3. Cooler current is almost entirely explained by case temperature, about 25 mA per °C on both units (r = 0.997). The EDFA room is warmest at night.**

* Fitted slopes are 24.5 and 24.9 mA/°C, and the 44-hour hot spell sits on the same line (plot C3).
* The daily temperature swing is small (1.7–1.8 °C) and **inverted**: warmest 02:00–05:00 IST, coolest around 17:00 IST. The SFP site follows the normal pattern (laser temperatures peak 16:00–18:00 IST).

*Why it matters:* the cooler's health check should be "Itec given case temperature", never Itec alone. The inverted cycle suggests the EDFA room's air conditioning is scheduled down at night. That is plausible but unverified, and worth checking against the site's HVAC settings before summer.

**4. The two amplifiers sit in different operating classes, and that fully explains their different pump drive.**

| | 010147e2 | 7a1d09b1 |
|---|---|---|
| POUT set-point | 15.0 dBm | 7.1 dBm |
| Gain | 14.8 dB | 5.4 dB |
| Pump current | 167 mA | 67 mA |

* Output power differs 6.2× while pump current differs only 2.5×.
* Both have their POUT alarm 3 dB below the set-point, and 9.2 and 10.7 dB of PIN margin above the −9 dBm input alarm.
* PIN never moved more than 0.3 dB on either unit, so no upstream fibre event occurred in the data.

*Why it matters:* scaling, thresholds and spare-part planning must be done per operating class. A shared threshold would mark 010147e2 as permanently "over-driven".

**5. On 7a1d09b1 the reported pump power fell 28% (26.0 → 18.6 mW, in two steps on 22 and 24 Feb), with pump current (66.8 mA), POUT (7.1 dBm) and case temperature unchanged.**

* By the primer, falling pump power at constant current is the pump-degradation signature.
* But a real 28% loss of pump light would cut the gain, so POUT would sag or the control loop would raise the current. Neither happened.

*Why it matters:* the physics points to the **pump-power monitor or its reporting path**, not the pump. Recommended action: verify pump power locally before scheduling a pump swap. Treating it as pump end-of-life would replace a working amplifier.

**6. Three EDFA "events" carry a monitoring signature: they start exactly at 00:00 UTC and introduce values the sensor never produced before.**

* **010147e2 pump current** reads 162.67 mA, −2.7%, from 00:00 UTC on 25 Mar (on and off until the end of the data). Before that the sensor only ever reported values on a 0.1 mA grid, such as 166.4 and 167.2.
* **7a1d09b1 pump power** steps fall at 00:00 on 22 and 24 Feb, to exactly 85% and 72% of 26.0 mW.
* **7a1d09b1 pump temperature** jumps to a fixed 34.681 °C, 9.7 °C above its set-point, from 00:00 on 28 Feb and again on 1 Mar. Meanwhile the cooler current does not react (194 mA during these readings, against 178 mA that week). A pump really 9.7 °C over its set-point would drive the TEC to its limit.

*Why it matters:* this separates what the devices did from what the monitoring did. Real pump changes are not scheduled for UTC midnight and do not change a sensor's resolution. These are processing artefacts, or deliberately injected test anomalies (the delivered files are named `*_anomalies_f.csv`). Either way, they should be routed to the monitoring team, not to field maintenance. The detector in Part D flags all three anyway.

**7. The four SFP reporting windows disagree. Every step change in the SFP data is confined to a single window.**

* **`Current Day`** carries almost all impossible values: 23.8% of its Laser Voltage readings, plus negative bias and Tx.
* **`Previous 15min`** is stamped 49–94 minutes **ahead** of the poll time, because the module clock is fast. Its bias also sits 5.3 mA above every other window on 1-4-3 throughout.
* **`Current`** steps on its own:
  * Tx on 1-4-3 −12% (0.83 → 0.73 mW) on 6 Jan;
  * bias on 1-4-4 +5.5 mA (+16%) on 18 Feb;
  * bias on 1-4-3 −4.4 mA on 27 Feb.

  In all three cases the day windows, computed by the same module, show no step.
* Status flags (present, operational, interval valid) were constant for all 88 days, so there were no port flaps.

*Why it matters:* a single-window step is a reporting-path problem. A laser that really aged 16% would move every window. Alerting on any one window would send technicians to swap healthy optics. Comparing windows against each other is a cheap, label-free integrity check.

**8. Both SFP lasers warmed by about 8 °C between early January and late March (46.6 → 55.0 °C and 45.6 → 53.9 °C), tracking the shelf temperature.**

* One of the unmapped ids (`3d9bf6d0`) behaves like a chassis temperature: 42 → 49 °C, correlating with the laser temperatures at r = 0.81 and 0.92.
* Laser bias in the day windows rose about 2.5 mA, consistent with the fitted 0.29–0.31 mA/°C thermal coefficient. That is normal physics, not ageing.
* At the observed 0.73 °C/week, late-March daily peaks (58.1 and 56.9 °C) reach 70 °C in about 16–18 weeks, that is by mid-July. 70 °C is an *assumed* commercial-grade SFP limit; the part numbers are unknown.
* Two other unmapped ids behave like the shelf's −48 V DC feeds (−46.8 to −48.1 V), sitting well inside the constant −43.2 / −56 V values that look like their alarm limits.

*Why it matters:* the Indian summer (April–June) will push these lasers toward their thermal limit before any ageing does. Check the shelf's fans and cooling at this site now, and confirm the SFP temperature grade.
