# Remote Sensing Practice

A collection of small, self-contained practice projects around satellite data and machine learning. Each topic lives in its own folder and new ones are added regularly.

| Folder | Topic | Status |
|---|---|---|
| [`Cloud Prediction/`](Cloud%20Prediction) | 1-hour cloud nowcasting from MSG satellite cloud masks | Done |

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
```

Reading the raw GRIB files additionally needs the [ECCODES](https://github.com/ecmwf/eccodes) library (required by `pygrib`).

**Note on data:** the raw satellite data is several hundred GB and is **not** part of this repository (`data/` is git-ignored). Each practice describes how to obtain or regenerate what it needs.

---

## Cloud Prediction

**Question:** can we predict whether a single site will be cloudy one hour from now, using only the surrounding satellite cloud mask?

**Data:** EUMETSAT MSG cloud mask (`EO:EUM:DAT:MSG:CLM`), 15-minute resolution, May and June 2025. A 41 × 41 pixel window (roughly 130 × 230 km) is cut out around the site.

**Approach:** start from a persistence baseline ("it stays as it is") and add features step by step: cloud fraction in rings around the site, directional sectors, trends and time of day. Logistic regression, random forest and gradient boosting are compared with a chronological train/test split.

**Main findings**
- The best models beat persistence by about 4 percentage points at a 1-hour lead time.
- Random forest and gradient boosting are no better than logistic regression, so the features are the limiting factor, not the model.
- The advantage is largest around 1 h, when clouds from within roughly 10-60 km arrive at the site. Beyond that the 41 × 41 window is too small.

### Notebooks

| Notebook | Purpose |
|---|---|
| [`prepare data.ipynb`](Cloud%20Prediction/prepare%20data.ipynb) | Reads the downloaded GRIB files and stores the site window as `data/cloud-prediction/clm_site.nc` |
| [`analyze data.ipynb`](Cloud%20Prediction/analyze%20data.ipynb) | Feature engineering, modelling and evaluation |

### Reproducing the data

**1. Credentials.** Create an account at EUMETSAT and put the API key and secret into a `.env` file in the repository root (git-ignored):

```
EUMETSAT_KEY=...
EUMETSAT_SECRET=...
```

**2. Download.** The download of roughly 5000 GRIB files is slow, so it ran overnight in a Docker container on a NAS with the [`eumdac`](https://pypi.org/project/eumdac/) command-line tool (`docker-compose.yml` and the full command are documented in `prepare data.ipynb`). The same download can be run locally with [`download.bash`](Cloud%20Prediction/download.bash):

```bash
uv run eumdac download \
    --collection EO:EUM:DAT:MSG:CLM \
    --start 2025-05-01T00:00:00 --end 2025-06-30T23:59:59 \
    -o "data/cloud-prediction/2025-05 to 06" -t 3 --entry '*.grb' --yes
```

**3. Extract the site window.** `prepare data.ipynb` cuts a 41 × 41 pixel window around the site (row `R`, column `C` in the full-disk grid) out of every file and stacks them into one time series:

```python
R, C = 3354, 1637

for file in files:
    with pg.open(file) as fn:
        date = fn[1].analDate
        subset = fn[1].values[R-20:R+21, C-20:C+21]
    ...

clm_subset = xr.DataArray(
    np.stack(subset_list),
    dims=("time", "y", "x"),
    coords={"time": np.array(date_list, dtype="datetime64[ns]")},
    name="clm",
).sortby("time")

clm_subset.to_netcdf("../data/cloud-prediction/clm_site.nc")
```

Afterwards run `analyze data.ipynb`.

### Scope and limitations
This is a practice project, kept deliberately small because downloading the satellite data takes very long. Results should be read with that in mind:
- Only two months (May and June 2025) are used, so the test set covers about 18 days and a single season.
- One site and one 41 × 41 pixel window are used, so the results may not transfer to other locations.
- Accuracy is the only metric, and the models use default hyperparameters without tuning.

### Possible next steps
- A larger window or wind / cloud-motion information (e.g. optical flow) to see clouds further upwind.
- More data covering other seasons.
- Metrics beyond accuracy, since the classes are imbalanced.
