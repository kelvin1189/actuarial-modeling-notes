## ============================================================
##  Article 1 figures and statistics
##  - GIF animation of TWD zero-coupon yield curves (1Y-10Y)
##  - Correlation matrices of levels and weekly changes
##  Input  : TPEx_ZCYC_Svensson2010010120251231.csv
##  Output : historical_yield_animation.gif
##           slope_timeseries.png
##           corr_levels.csv, corr_changes.csv
## ============================================================

graphics.off()
rm(list = ls())
cat("\014")

## ---- 0. packages ------------------------------------------------------
## Install on first run:
##   install.packages(c("dplyr", "tidyr", "lubridate", "readr",
##                      "ggplot2", "gganimate", "gifski", "scales"))

library(dplyr)
library(tidyr)
library(lubridate)
library(readr)
library(ggplot2)
library(gganimate)
library(gifski)
library(scales)

## ---- 1. paths ---------------------------------------------------------
wd_dir <- paste0(getwd(),"/")
setwd(wd_dir)

csv_path <- "TPEx_ZCYC_Svensson2010010120251231.csv"
PATH_OUT <- "."                                  # output directory
DATE_FROM <- as.Date("2010-01-08")
DATE_TO   <- as.Date("2025-12-31")
TENORS_YR <- 1:10  # ICS integer-year tenors used in this series

## ---- 2. load and sample -----------------------------------------------
##  CSV dates are in d/m/Y format; the column names contain spaces, so we
##  read with check.names = FALSE and rename to numeric tenor labels.

# TPEx CSV: dates are DD/MM/YYYY, yields stored as decimals (0.01 = 1%)
raw <- read_csv(csv_path, show_col_types = FALSE) |>
  mutate(Date = dmy(Date))

raw$Date <- as.Date(raw$Date, format = "%d/%m/%Y")

# Keep only the 1Y..10Y integer-year columns
yr_cols <- paste0(TENORS_YR, " year")
stopifnot(all(yr_cols %in% names(raw)))

daily <- raw %>%
  select(Date, all_of(yr_cols)) %>%
  filter(Date >= DATE_FROM, Date <= DATE_TO) %>%
  arrange(Date)

# Weekly sample: last available trading day in each ISO calendar week
weekly <- daily %>%
  mutate(iso_year = isoyear(Date), iso_week = isoweek(Date)) %>%
  group_by(iso_year, iso_week) %>%
  slice_tail(n = 1) %>%
  ungroup() %>%
  arrange(Date) %>%
  select(-iso_year, -iso_week)

cat(sprintf("Weekly observations: %d\n", nrow(weekly)))
cat(sprintf("Date range          : %s to %s\n",
            min(weekly$Date), max(weekly$Date)))

## ---- 3. tidy long format for plotting ---------------------------------
long <- weekly %>%
  pivot_longer(cols = all_of(yr_cols),
               names_to = "tenor_lbl",
               values_to = "yield") %>%
  mutate(tenor = as.numeric(sub(" year", "", tenor_lbl)),
         yield_pct = yield * 100)

# Slope (10Y - 1Y) per week, used to colour each frame's curve
slope_by_week <- weekly %>%
  mutate(slope_bp = (`10 year` - `1 year`) * 1e4) %>%
  select(Date, slope_bp)

long <- long %>% left_join(slope_by_week, by = "Date")

## ---- 4. y-axis range and colour scale (fixed across frames) -----------
y_min <- floor(min(long$yield_pct) * 10) / 10   # 0.1% grid
y_max <- ceiling(max(long$yield_pct) * 10) / 10

slope_lim <- range(long$slope_bp)

## ---- 5. animation -----------------------------------------------------
p <- ggplot(long, aes(x = tenor, y = yield_pct, colour = slope_bp,
                      group = Date)) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 2.2) +
  scale_x_continuous(breaks = TENORS_YR,
                     labels = paste0(TENORS_YR, "Y")) +
  scale_y_continuous(limits = c(y_min, y_max),
                     labels = label_number(accuracy = 0.1, suffix = "%")) +
  scale_colour_gradient2(
    low = "#2166AC", mid = "#969696", high = "#B2182B",
    midpoint = mean(slope_lim),
    limits = slope_lim,
    name = "10Y - 1Y\nspread (bp)"
  ) +
  labs(
    title = "TWD zero-coupon yield curve - {format(frame_time, '%Y-%m-%d')}",
    subtitle = "Weekly Svensson curve at ICS tenors (1Y-10Y)",
    x = "Maturity",
    y = "Zero-coupon yield",
    caption = "Source: derived from TPEx Treasury Yield Curve, Svensson methodology."
  ) +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank()) +
  transition_time(Date) +
  ease_aes("linear")

# Rendering parameters: 824 frames is fine at 20 fps (~41s clip).
# Width/height/res sized for a blog post.
anim <- animate(
  p,
  nframes  = nrow(weekly),
  fps      = 20,
  width    = 900,
  height   = 540,
  res      = 110,
  renderer = gifski_renderer()
)

anim_save(file.path(PATH_OUT, "historical_yield_animation.gif"), animation = anim)

## ---- 6. slope time series (PNG) ---------------------------------------
slope_plot <- ggplot(slope_by_week, aes(x = Date, y = slope_bp)) +
  geom_hline(yintercept = 0, colour = "grey50", linewidth = 0.4) +
  geom_line(colour = "#2166AC", linewidth = 0.5) +
  scale_x_date(date_breaks = "2 years", date_labels = "%Y") +
  labs(
    title = "TWD curve slope: 10-year minus 1-year spread",
    subtitle = "Weekly observations, 2010-01-08 to 2025-12-31",
    x = NULL,
    y = "Spread (basis points)",
    caption = "Source: derived from TPEx Treasury Yield Curve, Svensson methodology."
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave(file.path(PATH_OUT, "slope_timeseries.png"),
       slope_plot, width = 9, height = 4.5, dpi = 150)

## ---- 7. correlation matrices ------------------------------------------
Y  <- as.matrix(weekly[, yr_cols])             # 824 x 10 levels
dY <- diff(Y)                                  # 823 x 10 weekly changes

corr_levels  <- cor(Y)
corr_changes <- cor(dY)

# Pretty labels
dimnames(corr_levels)  <- list(yr_cols, yr_cols)
dimnames(corr_changes) <- list(yr_cols, yr_cols)

cat("\n=== Correlation of yield LEVELS ===\n")
print(round(corr_levels, 3))

cat("\n=== Correlation of weekly CHANGES ===\n")
print(round(corr_changes, 3))

# Save to CSV for the article
write.csv(round(corr_levels,  4),
          file.path(PATH_OUT, "corr_levels.csv"))
write.csv(round(corr_changes, 4),
          file.path(PATH_OUT, "corr_changes.csv"))

## ---- 8. summary diagnostic --------------------------------------------
off_diag <- function(M) M[lower.tri(M)]

cat(sprintf("\nLevels  off-diagonal correlation: mean %.3f, min %.3f\n",
            mean(off_diag(corr_levels)),  min(off_diag(corr_levels))))
cat(sprintf("Changes off-diagonal correlation: mean %.3f, min %.3f\n",
            mean(off_diag(corr_changes)), min(off_diag(corr_changes))))
cat(sprintf("\n1Y-10Y correlation: levels %.3f, changes %.3f\n",
            corr_levels["1 year", "10 year"],
            corr_changes["1 year", "10 year"]))
