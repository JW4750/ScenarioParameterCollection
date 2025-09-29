# HighD Hazardous Scenario Detection

This module extends the Erwin de Gelder highway scenario catalogue with
an additional safety screening step that looks for hazardous car-following
situations directly in the HighD trajectory data. The detector evaluates
every frame for longitudinal conflicts using established safety metrics:

- **Time-to-collision (TTC)**. Minderhoud and Bovy show that road users
  typically regard TTC values below 1.5 s as critical, especially in
  motorway settings where reaction times are limited.【F:docs/hazard_detection.md†L9-L12】
- **Time headway (THW)**. The Euro NCAP and German road-safety
  investigations often flag THW below 0.8 s as unsafe following behaviour
  that leaves little reaction margin.【F:docs/hazard_detection.md†L12-L15】
- **Distance headway (DHW)**. Empirical studies on German highways
  recommend maintaining at least a ten metre gap for passenger cars; gaps
  shorter than that, combined with closing speed, are linked to elevated
  rear-end crash risk.【F:docs/hazard_detection.md†L15-L18】

A frame is marked as hazardous when a lead vehicle is present and at least
one of the thresholds is violated (with the DHW rule additionally requiring
a positive closing speed). The HighD dataset does not provide intersecting
paths that would allow a reliable Post-Encroachment Time (PET) calculation,
so PET screening is out of scope for now; the logic focuses on the
longitudinal conflicts for which HighD already publishes TTC/THW/DHW.

Hazardous frames are consolidated into events and cross-referenced with the
Erwin scenarios detected from tag combinations:

1. If the hazardous frames overlap with one of the ten Erwin scenarios, the
   incident is attributed to that scenario for context.
2. If no overlap exists, the event is labelled as an **unknown hazardous
   scenario** and exported separately. The detector provides the frame
   bounds, minimum TTC/THW/DHW and the reasons that triggered the alert.

For fleet-level monitoring the detector also integrates the travelled
kilometres of all vehicles and computes the average distance between unknown
hazardous events. This helps quantify how frequently novel high-risk
situations occur in the observed traffic flow.

**References**

- Minderhoud, M. M., & Bovy, P. H. L. (2001). Extended time-to-collision measures for road traffic safety assessment. *Accident Analysis & Prevention*, 33(1), 89–97.
- German Federal Highway Research Institute (BASt). (2010). *Safety distance analyses on German motorways*.
- Euro NCAP. (2023). *AEB Car-to-Car Test Protocol*.
