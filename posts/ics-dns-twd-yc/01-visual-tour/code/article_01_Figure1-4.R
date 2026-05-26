# =============================================================================
# Article 1: A Visual Tour of Taiwan's Yield Curve
# Reproduces all four figures plus the correlation matrices used in the article.
#
#   Figure 1: historical_yield_animation.gif  — animated weekly yield curves
#   Figure 2: mean_std_by_tenor.png           — bar chart of mean and std dev
#   Figure 3: slope_timeseries.png            — 10Y minus 1Y spread time series
#   Figure 4: y10_distribution.png            — 10Y rate histogram with KDE
#
#   Tables:   corr_levels.csv, corr_changes.csv
#
# Inputs : TPEx_ZCYC_Svensson2010010120251231.csv in working directory
#          (edit `csv_path` below if elsewhere)
# Outputs: 4 image files + 2 CSVs written to the working directory
#
# Required packages:
#   dplyr, tidyr, ggplot2, readr, lubridate, scales, gganimate, gifski
# Install once:
#   install.packages(c("tidyverse", "scales", "gganimate", "gifski"))
#
# Run time: ~30 seconds for the animation, < 5 seconds for everything else.
# If iterating on static figures only, comment out the Figure 1 block.
#
# Reproducibility: tested with R 4.4.2; sessionInfo() at end of file.
# =============================================================================

graphics.off()
rm(list = ls())
cat("\014")

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(lubridate)
  library(scales)
  library(ggplot2)
  library(gganimate)
  library(gifski)
})


# =============================================================================
# 1. Setup: paths and date range
# =============================================================================

wd_dir <- paste0(getwd(),"/")
setwd(wd_dir)


csv_path  <- "TPEx_ZCYC_Svensson2010010120251231.csv"
out_dir   <- "."
DATE_FROM <- as.Date("2010-01-08")
DATE_TO   <- as.Date("2025-12-31")
TENORS_YR <- 1:10                 # ICS integer-year tenors used in this series
yr_cols   <- paste0(TENORS_YR, " year")


# =============================================================================
# 2. Load and prepare the weekly sample
# =============================================================================

# TPEx CSV: dates are DD/MM/YYYY, yields stored as decimals (0.01 = 1%)
raw <- read_csv(csv_path, show_col_types = FALSE) |>
  mutate(Date = dmy(Date))

raw <- readRDS("TPEx_ZCYC_Svensson_Wide.rds")

stopifnot(all(yr_cols %in% names(raw)))

# Restrict to integer-year tenors and the analysis window
daily <- raw |>
  select(Date, all_of(yr_cols)) |>
  filter(Date >= DATE_FROM, Date <= DATE_TO) |>
  arrange(Date)

# Weekly sample: last available trading day in each ISO calendar week
weekly <- daily |>
  mutate(iso_year = isoyear(Date),
         iso_week = isoweek(Date)) |>
  group_by(iso_year, iso_week) |>
  slice_max(Date, n = 1) |>
  ungroup() |>
  arrange(Date) |>
  select(-iso_year, -iso_week)

stopifnot(nrow(weekly) == 824)   # tripwire: fails loudly if data drifts


# =============================================================================
# 3. Shared theme and color palette (used by Figures 2, 3, 4)
# =============================================================================

# Single theme function. Plots that don't want a legend override with
# `+ theme(legend.position = "none")` at the call site.
theme_article <- function() {
  theme_minimal(base_size = 12) +
    theme(
      panel.grid.major.x   = element_blank(),
      panel.grid.minor     = element_blank(),
      panel.grid.major.y   = element_line(linetype = "dotted",
                                          colour = "grey70",
                                          linewidth = 0.3),
      axis.line.x          = element_line(colour = "grey50", linewidth = 0.3),
      axis.line.y          = element_line(colour = "grey50", linewidth = 0.3),
      axis.ticks           = element_line(colour = "grey50", linewidth = 0.3),
      axis.text            = element_text(colour = "grey30"),
      axis.title           = element_text(colour = "grey20"),
      legend.position      = "top",
      legend.justification = "left",
      legend.title         = element_blank(),
      legend.key.size      = unit(0.9, "lines"),
      plot.title           = element_blank(),
      plot.margin          = margin(8, 12, 8, 8)
    )
}

# Site-consistent palette
col_primary   <- "#4a5d8f"   # dark blue: main lines, mean bars, histogram fill
col_secondary <- "#a8b8d8"   # light blue: std dev bars
col_accent    <- "#c44444"   # red: KDE overlay, zero reference line
col_highlight <- "#1c1f24"   # dark navy: annotation points and labels
col_reference <- "#333333"   # dark grey: median reference line

# Common output dimensions for static figures (Figures 2, 3, 4)
fig_width  <- 9
fig_height <- 5
fig_dpi    <- 140


# =============================================================================
# 4. Figure 1: animated weekly yield curves (GIF)
# =============================================================================

# Long format for plotting; one row per (Date, tenor)
long <- weekly |>
  pivot_longer(cols = all_of(yr_cols),
               names_to  = "tenor_lbl",
               values_to = "yield") |>
  mutate(tenor     = as.numeric(sub(" year", "", tenor_lbl)),
         yield_pct = yield * 100)

# Slope (10Y - 1Y) per week, used to colour each frame's curve
slope_by_week <- weekly |>
  mutate(slope_bp = (`10 year` - `1 year`) * 1e4) |>
  select(Date, slope_bp)

long <- long |>
  left_join(slope_by_week, by = "Date")

# Fixed y-axis range and colour-scale limits so frames are comparable
y_min     <- floor(min(long$yield_pct) * 10) / 10   # round down to 0.1%
y_max     <- ceiling(max(long$yield_pct) * 10) / 10
slope_lim <- range(long$slope_bp)

p_anim <- ggplot(long,
                 aes(x = tenor, y = yield_pct,
                     colour = slope_bp, group = Date)) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 2.2) +
  scale_x_continuous(breaks = TENORS_YR,
                     labels = paste0(TENORS_YR, "Y")) +
  scale_y_continuous(limits = c(y_min, y_max),
                     labels = label_number(accuracy = 0.1, suffix = "%")) +
  scale_colour_gradient2(
    low      = "#2166AC",
    mid      = "#969696",
    high     = "#B2182B",
    midpoint = mean(slope_lim),
    limits   = slope_lim,
    name     = "10Y - 1Y\nspread (bp)"
  ) +
  labs(
    title    = "TWD zero-coupon yield curve - {format(frame_time, '%Y-%m-%d')}",
    subtitle = "Weekly Svensson curve at ICS tenors (1Y-10Y)",
    x        = "Maturity",
    y        = "Zero-coupon yield",
    caption  = "Source: derived from TPEx Treasury Yield Curve, Svensson methodology."
  ) +
  theme_minimal(base_size = 13) +
  theme(plot.title       = element_text(face = "bold"),
        panel.grid.minor = element_blank()) +
  transition_time(Date) +
  ease_aes("linear")

# 824 frames at 20 fps = ~41-second GIF.
anim <- animate(
  p_anim,
  nframes  = nrow(weekly),
  fps      = 20,
  width    = 900,
  height   = 540,
  res      = 110,
  renderer = gifski_renderer()
)

anim_save(file.path(out_dir, "historical_yield_animation.gif"), animation = anim)


# =============================================================================
# 5. Figure 2: bar chart of mean and standard deviation by tenor
# =============================================================================

tenor_stats <- weekly |>
  select(all_of(yr_cols)) |>
  pivot_longer(everything(),
               names_to  = "tenor_label",
               values_to = "yield") |>
  mutate(tenor_years = as.integer(gsub(" year", "", tenor_label)),
         yield_pct   = yield * 100) |>
  group_by(tenor_years) |>
  summarise(mean_pct = mean(yield_pct),
            sd_pct   = sd(yield_pct),
            .groups  = "drop") |>
  arrange(tenor_years)

# Reshape to long format for grouped bars
tenor_long <- tenor_stats |>
  pivot_longer(c(mean_pct, sd_pct),
               names_to  = "stat",
               values_to = "value") |>
  mutate(stat = factor(stat,
                       levels = c("mean_pct", "sd_pct"),
                       labels = c("Mean yield (%)",
                                  "Standard deviation (%)")))

p_bars <- ggplot(tenor_long,
                 aes(x = factor(tenor_years), y = value, fill = stat)) +
  geom_col(position = position_dodge(width = 0.78), width = 0.72) +
  geom_text(aes(label = sprintf("%.2f", value), colour = stat),
            position    = position_dodge(width = 0.78),
            vjust       = -0.5,
            size        = 3.1,
            show.legend = FALSE) +
  scale_fill_manual(values = c("Mean yield (%)"         = col_primary,
                               "Standard deviation (%)" = col_secondary)) +
  scale_colour_manual(values = c("Mean yield (%)"         = "grey20",
                                 "Standard deviation (%)" = "grey40")) +
  scale_x_discrete(labels = function(x) paste0(x, "Y")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.10))) +
  labs(x = "Tenor (years)", y = "Yield (%)") +
  theme_article()

ggsave(file.path(out_dir, "mean_std_by_tenor.png"), plot = p_bars,
       width = fig_width, height = fig_height,
       dpi = fig_dpi, bg = "white")


# =============================================================================
# 6. Figure 3: slope time series with annotated highlights
# =============================================================================

slope_data <- weekly |>
  mutate(spread_bp = (`10 year` - `1 year`) * 10000) |>   # decimal -> bp
  select(Date, spread_bp)

slope_summary <- slope_data |>
  summarise(mean_bp    = mean(spread_bp),
            median_bp  = median(spread_bp),
            sd_bp      = sd(spread_bp),
            max_bp     = max(spread_bp),
            min_bp     = min(spread_bp),
            n_negative = sum(spread_bp < 0))

# Highlighted dates. Per-label hjust avoids clipping at the plot edges:
#   - leftmost label (2010-01-29): hjust = 0 anchors at the point, extends right
#   - interior labels:              hjust = 0.5 (centered above/below point)
highlight_dates <- tribble(
  ~date,                 ~label,                        ~vjust_adj, ~hjust_adj,
  as.Date("2010-01-29"), "Steepest curve\n(175 bp)",    -0.8,        0,
  as.Date("2020-11-27"), "Lowest 10Y rate\n(0.26%)",     1.6,        0.5,
  as.Date("2022-10-21"), "Highest 10Y rate\n(1.91%)",   -0.8,        0.5
) |>
  left_join(slope_data, by = c("date" = "Date"))

p_slope <- ggplot(slope_data, aes(x = Date, y = spread_bp)) +
  geom_hline(yintercept = 0,
             colour     = col_accent,
             linetype   = "dashed",
             linewidth  = 0.4,
             alpha      = 0.7) +
  geom_line(colour = col_primary, linewidth = 0.45, alpha = 0.9) +
  geom_point(data = highlight_dates,
             aes(x = date, y = spread_bp),
             colour = col_highlight,
             size   = 2.2) +
  geom_text(data = highlight_dates,
            aes(x = date, y = spread_bp, label = label,
                vjust = vjust_adj, hjust = hjust_adj),
            size       = 3.1,
            colour     = col_highlight,
            lineheight = 0.95) +
  scale_x_date(
    breaks = seq(as.Date("2010-01-01"), as.Date("2026-01-01"), by = "2 years"),
    labels = date_format("%Y"),
    expand = expansion(mult = c(0.02, 0.02))
  ) +
  scale_y_continuous(
    breaks = seq(-25, 200, by = 25),
    labels = function(b) paste0(b, " bp"),
    expand = expansion(mult = c(0.10, 0.10))
  ) +
  labs(x = NULL, y = "10Y - 1Y spread (basis points)") +
  theme_article() +
  theme(legend.position = "none")

ggsave(file.path(out_dir, "slope_timeseries.png"), plot = p_slope,
       width = 9, height = 7,
       dpi = 150, bg = "white")


# =============================================================================
# 7. Figure 4: 10-year rate distribution with KDE overlay
# =============================================================================

y10 <- weekly |>
  transmute(y10_pct = `10 year` * 100)

med_y10   <- median(y10$y10_pct)
bin_width <- diff(range(y10$y10_pct)) / 40   # target ~40 bins across range

p_hist <- ggplot(y10, aes(x = y10_pct)) +
  geom_histogram(binwidth  = bin_width,
                 fill      = col_primary,
                 colour    = "white",
                 linewidth = 0.4,
                 alpha     = 0.85) +
  # Scale density to count axis: density * n * bin_width
  geom_density(aes(y = after_stat(count) * bin_width),
               colour    = col_accent,
               linewidth = 0.9,
               bw        = "nrd0") +
  geom_vline(xintercept = med_y10,
             linetype   = "dashed",
             colour     = col_reference,
             linewidth  = 0.5,
             alpha      = 0.7) +
  annotate("text",
           x = med_y10 + 0.02, y = Inf,
           label = sprintf("Median: %.2f%%", med_y10),
           hjust = 0, vjust = 1.5,
           size = 3.4, colour = col_reference) +
  annotate("text",
           x = min(y10$y10_pct), y = Inf,
           label = "Kernel density estimate (red)",
           hjust = 0, vjust = 3.0,
           size = 3.4, colour = col_accent) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.06))) +
  labs(x = "10-year zero-coupon rate (%)",
       y = "Number of weekly observations") +
  theme_article() +
  theme(legend.position = "none")

ggsave(file.path(out_dir, "y10_distribution.png"), plot = p_hist,
       width = fig_width, height = fig_height,
       dpi = fig_dpi, bg = "white")


# =============================================================================
# 8. Correlation matrices (saved as CSV for the article tables)
# =============================================================================

Y  <- as.matrix(weekly[, yr_cols])   # 824 x 10 levels
dY <- diff(Y)                        # 823 x 10 weekly changes

corr_levels  <- cor(Y)
corr_changes <- cor(dY)

dimnames(corr_levels)  <- list(yr_cols, yr_cols)
dimnames(corr_changes) <- list(yr_cols, yr_cols)

write.csv(round(corr_levels,  4),
          file.path(out_dir, "corr_levels.csv"))
write.csv(round(corr_changes, 4),
          file.path(out_dir, "corr_changes.csv"))


# =============================================================================
# 9. Consolidated sanity statistics
# =============================================================================

cat("\n========================================\n")
cat("  Sanity check: numbers in the article\n")
cat("========================================\n\n")

cat(sprintf("Weekly observations: %d (expected 824)\n", nrow(weekly)))
cat(sprintf("Date range: %s to %s\n\n",
            format(min(weekly$Date)), format(max(weekly$Date))))

cat("--- 10Y rate distribution ---\n")
cat(sprintf("Min:    %.3f%%    Median: %.3f%%    Mean: %.3f%%    Max: %.3f%%\n",
            min(y10$y10_pct), med_y10, mean(y10$y10_pct), max(y10$y10_pct)))
cat(sprintf("Share below 1.0%%:           %5.1f%%\n",
            mean(y10$y10_pct < 1) * 100))
cat(sprintf("Share in [1.0%%, 1.5%%):     %5.1f%%\n",
            mean(y10$y10_pct >= 1 & y10$y10_pct < 1.5) * 100))
cat(sprintf("Share at or above 1.5%%:    %5.1f%%\n\n",
            mean(y10$y10_pct >= 1.5) * 100))

cat("--- Slope (10Y - 1Y) summary ---\n")
cat(sprintf("Mean:           %5.1f bp     (article: 51.4)\n",  slope_summary$mean_bp))
cat(sprintf("Median:         %5.1f bp     (article: 49.3)\n",  slope_summary$median_bp))
cat(sprintf("Std dev:        %5.1f bp     (article: 33.2)\n",  slope_summary$sd_bp))
cat(sprintf("Maximum:        %5.1f bp     (article: 175.5)\n", slope_summary$max_bp))
cat(sprintf("Minimum:        %5.1f bp     (article: -5.6)\n",  slope_summary$min_bp))
cat(sprintf("Negative weeks: %d of %d  (article: 3)\n\n",
            slope_summary$n_negative, nrow(slope_data)))

cat("--- Coefficient of variation by tenor ---\n")
tenor_stats |>
  mutate(cov = sd_pct / mean_pct) |>
  mutate(across(c(mean_pct, sd_pct), \(x) sprintf("%.3f%%", x)),
         cov = sprintf("%.3f", cov)) |>
  print(n = 10)

cat("\n--- Correlation matrix off-diagonal summary ---\n")
off_diag <- function(M) M[lower.tri(M)]
cat(sprintf("Levels  off-diagonal: mean %.3f, min %.3f\n",
            mean(off_diag(corr_levels)),  min(off_diag(corr_levels))))
cat(sprintf("Changes off-diagonal: mean %.3f, min %.3f\n",
            mean(off_diag(corr_changes)), min(off_diag(corr_changes))))
cat(sprintf("1Y-10Y correlation: levels %.3f, changes %.3f\n",
            corr_levels["1 year",  "10 year"],
            corr_changes["1 year", "10 year"]))

cat("\n--- Files written ---\n")
cat("  Figures: historical_yield_animation.gif\n")
cat("           mean_std_by_tenor.png\n")
cat("           slope_timeseries.png\n")
cat("           y10_distribution.png\n")
cat("  Tables:  corr_levels.csv\n")
cat("           corr_changes.csv\n\n")


# =============================================================================
# sessionInfo() — uncomment to capture for reproducibility documentation
# =============================================================================
# sink("sessionInfo.txt"); print(sessionInfo()); sink()
