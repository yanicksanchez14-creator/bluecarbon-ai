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
image: for every pixel, the median of the clearest 20% of its cloud-free observations of the period,
ranked by near-infrared reflectance (least sun glint, haze and white water). A single clearest day is noisy;
the median of several clear days is stable while still letting a shallow seafloor show through. Its blue, green, red and NIR bands, plus the
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
| Seagrass | Allen Coral Atlas benthic map (tropics), plus official survey maps where they exist: the FWC Florida statewide seagrass map (surveys from 2004 on) and Seamap Australia (Moreton Bay 2015, Great Barrier Reef, Western Australia, Victoria 2023), burned in over water only, with unlabelled water within 1.5 km of a surveyed meadow taught as open water. Where seagrass exists but no map covers it, water outside the Atlas footprint is left unlabelled rather than taught as open water |

Pixels within 1 px of a class boundary are ignored, because edges are where global products are
least reliable.

**Splitting.** Chips don't overlap. Whole 5 km blocks go to train, val or test, and five estuaries
(Mission Bay, Plum Island, Moreton Bay, Shoalwater Bay, Tampa Bay) are held out entirely. Extra years of
imagery (2020, 2022) of the training sites go to training only. That tests geographic generalization, not memorization
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

### Training data added in October 2026 (labels v10)

The current model was trained on 71 sites (up from 39) with official survey maps as answer keys. It
replaced the previous model because it scored better on the five held-out estuaries, graded on the
same labels: mean blue carbon IoU 0.50 against 0.43, with no habitat at any estuary dropping by more
than 0.01. The Environment Agency (England) and NWI mangrove services could not be reached during
this run, so those sites used the global maps; they are retried in the next run.

- **Salt marsh:** US Fish and Wildlife Service National Wetlands Inventory (estuarine intertidal
  emergent), Environment Agency saltmarsh extents for England (aerial survey 2016-2019) and NSW
  estuarine macrophytes. Inside a survey's coverage it is taken as the complete map: where the global
  maps call a pixel salt marsh and the survey does not, the pixel is left out of training and scoring.
- **Mangrove:** labelled only where ESA WorldCover and Global Mangrove Watch v3 (Bunting et al. 2022)
  agree, plus the National Wetlands Inventory and NSW surveys.
- **Seagrass:** Virginia Institute of Marine Science annual SAV survey (2020-2024), the NOAA and Texas
  Parks and Wildlife Texas Benthic Atlas, MassDEP eelgrass survey 2019-2023, the California eelgrass
  compilation, NSW estuarine macrophytes and Seamap Australia Hervey Bay.
- **New sites:** Texas, Chesapeake and Virginia coastal bays, Massachusetts, California, Washington,
  New Jersey, South Carolina, New Hampshire and Florida in the US; four NSW estuaries, the Great Sandy
  Strait and Exmouth Gulf in Australia; three English salt marshes; and mangrove coasts in Kenya,
  Tanzania, Malaysia, Indonesia, Brazil, Ecuador, Honduras and Nigeria.

### Next run (code of October 12 2026, labels v11)

- **New inputs.** ALOS PALSAR-2 L-band radar (HH, HV and their ratio), which passes through the canopy to trunks and
  roots; and a satellite-derived relative depth, the Stumpf et al. (2003) log-ratio of the clear-water blue and green
  bands.
- **Seagrass is learned only where the seafloor is visible.** Many official seagrass surveys were mapped by boat or
  diver in water no satellite can see into. During training only, seagrass labels in optically deep water (blue and
  green within 0.004 reflectance of the darkest 2% of the tile's water) or sediment-laden water (red at least 0.04 and
  at least 0.85 x green) are skipped, never relabelled as water. Held-out scores still use every surveyed hectare.
  The share skipped at each training site is written to `seafloor_mask.json` in the results.
- **Mangrove biomass check.** Besides GEDI L4A footprints and the 10 m canopy height map, each mangrove site gets
  Meta's 1 m canopy height and a local height-to-biomass model (AGB = a x H^b) fitted to its own GEDI footprints,
  with a cross-validated error.
- **More answer keys and sites (95 sites).** Barnegat Bay SAV 2009 (Rutgers / NJDEP), Great Bay eelgrass 2022
  (PREP / UNH), EMODnet Seabed Habitats seagrass EOV 2023 (Europe), Torres Strait intertidal seagrass 2002-2014,
  Natural Resources Wales saltmarsh extents, the Saltmarsh Survey of Scotland and Ireland's Saltmarsh Monitoring
  Project. New sites in Spain, England, Germany, Torres Strait, Wales, Scotland, Ireland, the Netherlands, the Bay of
  Fundy, China, India, Myanmar, Madagascar, Senegal, Colombia and Panama.
- **Seven held-out estuaries**, adding Brisbane Water (New South Wales: seagrass, mangrove, salt marsh) and the
  Ribble Estuary (England: salt marsh).
- **Spot-check.** 200 points per held-out estuary (`spotcheck.csv`), half random and half on mapped blue carbon
  habitat, each with the model's class, the reference label and a link to high-resolution imagery, for a check by eye.

### Limitations
- Seagrass is the weakest habitat (0.46 IoU): it is spectrally hard to separate from water, and meadows in
  murky or deep water are missed. Unseen Tampa Bay scores 0.69, Moreton Bay 0.12 and Shoalwater Bay 0.18.
  Global seagrass labels only cover tropical reefs; temperate seagrass needs local survey polygons.
- Area correction (Olofsson et al.) uses the model's accuracy pooled over all test areas. On its own that can
  move a large share of a site's open water into seagrass that the map never found, so for mangrove, salt
  marsh and seagrass the corrected area is capped at mapped area x precision / recall. This is conservative
  (carbon is never claimed for habitat the map did not see) and under-states seagrass where it is badly
  under-mapped.
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
