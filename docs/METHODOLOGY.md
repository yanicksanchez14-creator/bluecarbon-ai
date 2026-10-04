## Pipeline

```
Sentinel-2 L2A ──► Cloud Score+ mask ──► seasonal median ──► 10 bands + 4 indices + elevation / tide ─┐
                                                                                  ├─► U-Net ──► habitat map ──► area ──► carbon
ESA WorldCover ─┬─► fused reference labels ──► boundary buffer ──► chips ─────────┘      (±TTA)     (error-      (Tier 1,
Murray tidal    │                                                  (spatial-block split)            adjusted)    Monte Carlo)
Allen Coral Atlas┘
```

**Imagery.** Sentinel-2 surface reflectance (`COPERNICUS/S2_SR_HARMONIZED`), masked with Google's
Cloud Score+ (`cs_cdf ≥ 0.6`), reduced to a per-pixel median over the chosen season. Bands B2–B8A,
B11, B12 at 10 m in the local UTM zone. The model also gets NDVI, NDWI, MNDWI and NDMI.

**Clear-water image.** Seagrass is visible from space only where the seafloor shows through, and a
yearly median blends clear days with murky, glinty ones. Each site therefore also gets a clear-water
image: for every pixel, the single cloud-free observation of the period with the lowest near-infrared
reflectance (least sun glint, haze and white water). Its blue, green, red and NIR bands, plus the
log band ratios ln(B2/B3) and ln(B3/B4) (largely insensitive to water depth, after Lyzenga and Stumpf),
are model inputs.

**Classes.** Open water · mangrove · salt marsh · seagrass · tidal flat · other land. `255` means
*no label* and is never trained on or scored.

**Reference labels.** These are fused from published global products rather than drawn by hand
or made with index thresholds:

| Class | Source |
|---|---|
| Water, other land | ESA WorldCover 2021 (10 m) |
| Mangrove | ESA WorldCover class 95 |
| Salt marsh | WorldCover herbaceous, grass or shrub cover that the GWL_FCS30 wetland map (Zhang et al. 2023) classes as salt marsh. Tidal-zone vegetation it calls non-wetland is left unlabelled |
| Freshwater wetland | WorldCover herbaceous wetland outside the tidal zone, or that GWL_FCS30 calls swamp / marsh |
| Tidal flat | Murray et al. global intertidal change |
| Seagrass | Allen Coral Atlas benthic map (tropics), plus official survey maps where they exist: the FWC Florida statewide seagrass map (surveys from 2010 on) and the 2015 Moreton Bay seagrass map (Seamap Australia), burned in over water only. At sites with seagrass that no map covers (e.g. Florida Bay, Tampa Bay), water outside the Atlas footprint is left unlabelled rather than taught as open water |

Pixels within 1 px of a class boundary are ignored, because edges are where global products are
least reliable.

**Splitting.** Chips don't overlap. Whole 5 km blocks go to train, val or test, and two sites
(Moreton Bay, Tampa Bay) are held out entirely. That tests geographic generalization, not memorization
of neighbouring pixels.

**Context layers.** Elevation (NASADEM) and tidal-wetland probability (Murray et al. 2022) are added as
inputs, because tidal vs. freshwater marsh can't be separated from one image alone. Both also feed the label
rules, so held-out estuary scores are the fair measure. Latitude is deliberately *not* an input: in an earlier
run the model used it as a shortcut ("no mangrove north of 25°") and missed the mangroves of held-out Tampa Bay
and Moreton Bay, even though its pooled test score looked good. Instead, the known global range of mangroves
(39°S–32.5°N) is applied as an explicit rule, and every run reports per-estuary scores for the held-out sites.

**Model.** A U-Net with a ResNet-34 encoder from `segmentation-models-pytorch`, a 17-channel input
stem, and cross-entropy plus Dice loss with square-root inverse-frequency class weights.
Training uses AdamW with one-cycle LR, mixed precision and early stopping on validation mIoU.
Inference uses overlapping tiles blended with a Hann window, plus flip test-time augmentation.

**Metrics.** Per-class IoU, F1, precision and recall, plus mIoU, macro-F1, overall accuracy and
Cohen's κ, all computed on held-out test chips. Overall accuracy is reported but never headlined:
a map that is 70% water can score 90% accuracy while missing every salt marsh pixel.

**Area estimation.** Mapped pixel counts are biased. Reported areas are *error-adjusted* with the
stratified estimator of Olofsson et al. (2014), using the model's held-out confusion matrix. The
confidence intervals treat pixels as independent samples, so they understate the true uncertainty.
A design-based accuracy assessment with independent reference points would tighten this.

**Carbon.** Soil organic carbon to 1 m comes from measured soil cores where possible: the Smithsonian
Coastal Carbon Library (v1.7.0) is reduced by `scripts/build_soil_carbon.py` to one stock per core (dry bulk
density × organic carbon fraction, averaged over at least 30 cm of samples and scaled to 1 m). Measured carbon
fractions are used only when the study confirms organic carbon (or carbonate removal); total-carbon values,
which include carbonate in tropical seafloor, are replaced by an estimate from organic matter (Craft et al.,
1991). Each site uses the cores of each habitat within 100 km (300 km if needed) when there are at least 8;
the coefficient's mean is the winsorized mean of those cores and its range reflects the number of independent
studies. Elsewhere, and for living biomass and soil carbon accumulation, IPCC 2013 Wetlands Supplement Tier 1
defaults are used. Each coefficient is sampled from a triangular
distribution (5,000 Monte Carlo draws), together with the area uncertainty, and results are reported
as the mean and 90% interval. The indicative credit value uses **annual sequestration only**, since
standing stock is not creditable. Tier 1 values are global averages; any real project needs
site-measured stocks.

### Limitations
- Seagrass is spectrally hard to separate from water, and global seagrass labels only cover tropical
  reefs. Temperate seagrass needs local survey polygons.
- Tides change what the satellite sees in intertidal zones. A median composite averages over tides.
- The global label products have their own errors, so the model learns those errors too.

### References
- IPCC (2014). *2013 Supplement to the 2006 IPCC Guidelines for National GHG Inventories: Wetlands*, Ch. 4.
- Olofsson, P. et al. (2014). Good practices for estimating area and assessing accuracy of land change. *RSE* 148.
- Zanaga, D. et al. (2022). ESA WorldCover 10 m 2021 v200.
- Murray, N. J. et al. (2022). High-resolution mapping of losses and gains of Earth's tidal wetlands. *Science* 376.
- Zhang, X. et al. (2023). GWL_FCS30: a global 30 m wetland map with a fine classification system. *ESSD* 15.
- Coastal Carbon Network (2023). Coastal Carbon Library v1.7.0. Smithsonian Environmental Research Center. doi:10.25573/serc.21565671.
- Craft, C. B. et al. (1991). Loss on ignition and Kjeldahl digestion for estimating organic carbon in estuarine marsh soils. *SSSAJ* 55.
- Allen Coral Atlas (2022). Imagery, maps and monitoring of the world's tropical coral reefs.
- Pasquarella, V. et al. (2023). Cloud Score+: comprehensive cloud and cloud-shadow detection for Sentinel-2.
