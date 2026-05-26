# =============================================================================
# Article 1: A Visual Tour of Taiwan's Yield Curve
# R script to reproduce the two new figures:
#   1. mean_std_by_tenor.png  — bar chart of mean and std dev by tenor
#   2. y10_distribution.png   — histogram of 10Y rate with KDE overlay
#
# Inputs : TPEx_ZCYC_Svensson2010010120251231.csv  (in working directory or
#          adjust `csv_path` below)
# Outputs: two PNG files written to the working directory
#
# Required packages: tidyverse (dplyr, tidyr, ggplot2, readr, lubridate)
# Install once:      install.packages("tidyverse")
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(readr)
  library(lubridate)
  library(ISOweek)   # for ISO-week handling; install.packages("ISOweek") if needed
})

# -----------------------------------------------------------------------------
# 1. Load and prepare the weekly sample
# -----------------------------------------------------------------------------

csv_path <- "TPEx_ZCYC_Svensson2010010120251231.csv"

# TPEx CSV: dates are DD/MM/YYYY, yields stored as decimals (0.01 = 1%)
raw <- read_csv(csv_path, show_col_types = FALSE) |>
  mutate(Date = dmy(Date))

# ISO-week sampling: last available trading day in each ISO calendar week.
# Restricted to 2010-01-08 (start of first complete ISO week of 2010)
# through 2025-12-31.
weekly <- raw |>
  filter(Date >= as.Date("2010-01-08"), Date <= as.Date("2025-12-31")) |>
  mutate(
    iso_year = isoyear(Date),
    iso_week = isoweek(Date)
  ) |>
  group_by(iso_year, iso_week) |>
  slice_max(Date, n = 1) |>
  ungroup() |>
  arrange(Date)

stopifnot(nrow(weekly) == 824)   # sanity check

# -----------------------------------------------------------------------------
# 2. Common theme — kept minimal so plots inherit cleanly across modes
# -----------------------------------------------------------------------------

theme_article <- function() {
  theme_minimal(base_size = 12) +
    theme(
      panel.grid.major.x = element_blank(),
      panel.grid.minor   = element_blank(),
      panel.grid.major.y = element_line(linetype = "dotted",
                                        colour = "grey70",
                                        linewidth = 0.3),
      axis.line.x        = element_line(colour = "grey50", linewidth = 0.3),
      axis.line.y        = element_line(colour = "grey50", linewidth = 0.3),
      axis.ticks         = element_line(colour = "grey50", linewidth = 0.3),
      axis.text          = element_text(colour = "grey30"),
      axis.title         = element_text(colour = "grey20"),
      legend.position    = "top",
      legend.justification = "left",
      legend.title       = element_blank(),
      legend.key.size    = unit(0.9, "lines"),
      plot.title         = element_blank(),
      plot.margin        = margin(8, 12, 8, 8)
    )
}

# Site-consistent blues
col_primary   <- "#4a5d8f"   # dark blue: means / histogram bars
col_secondary <- "#a8b8d8"   # light blue: std deviations
col_accent    <- "#c44444"   # red: KDE overlay
col_reference <- "#333333"   # dark grey: median line

# -----------------------------------------------------------------------------
# 3. Figure 1: bar chart of mean and standard deviation by tenor
# -----------------------------------------------------------------------------

# Compute mean and std dev for each integer tenor, store yields in percent
tenor_stats <- weekly |>
  select(matches("^[0-9]+ year$")) |>
  select(!matches("^[0-9]+\\.")) |>      # drop half-year tenors
  pivot_longer(everything(),
               names_to = "tenor_label",
               values_to = "yield") |>
  mutate(
    tenor_years = as.integer(gsub(" year", "", tenor_label)),
    yield_pct   = yield * 100
  ) |>
  filter(tenor_years >= 1, tenor_years <= 10) |>
  group_by(tenor_years) |>
  summarise(
    mean_pct = mean(yield_pct),
    sd_pct   = sd(yield_pct),
    .groups  = "drop"
  ) |>
  arrange(tenor_years)

# Reshape to long format for grouped bars
tenor_long <- tenor_stats |>
  pivot_longer(c(mean_pct, sd_pct),
               names_to  = "stat",
               values_to = "value") |>
  mutate(
    stat = factor(stat,
                  levels = c("mean_pct", "sd_pct"),
                  labels = c("Mean yield (%)", "Standard deviation (%)"))
  )

p1 <- ggplot(tenor_long,
             aes(x = factor(tenor_years), y = value, fill = stat)) +
  geom_col(position = position_dodge(width = 0.78), width = 0.72) +
  geom_text(aes(label = sprintf("%.2f", value),
                colour = stat),
            position = position_dodge(width = 0.78),
            vjust = -0.5,
            size = 3.1,
            show.legend = FALSE) +
  scale_fill_manual(values = c("Mean yield (%)"          = col_primary,
                               "Standard deviation (%)"  = col_secondary)) +
  scale_colour_manual(values = c("Mean yield (%)"          = "grey20",
                                 "Standard deviation (%)"  = "grey40")) +
  scale_x_discrete(labels = function(x) paste0(x, "Y")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.10))) +
  labs(x = "Tenor (years)", y = "Yield (%)") +
  theme_article()

ggsave("mean_std_by_tenor.png", plot = p1,
       width = 9, height = 5, dpi = 140, bg = "white")

# -----------------------------------------------------------------------------
# 4. Figure 2: 10-year rate distribution with KDE overlay
# -----------------------------------------------------------------------------

y10 <- weekly |>
  transmute(y10_pct = `10 year` * 100)

med_y10 <- median(y10$y10_pct)

# Bin width chosen to give ~40 bins across the observed range
bin_width <- diff(range(y10$y10_pct)) / 40

p2 <- ggplot(y10, aes(x = y10_pct)) +
  geom_histogram(
    binwidth = bin_width,
    fill = col_primary,
    colour = "white",
    linewidth = 0.4,
    alpha = 0.85
  ) +
  geom_density(
    aes(y = after_stat(count) * bin_width),
    colour = col_accent,
    linewidth = 0.9,
    bw = "nrd0"
  ) +
  geom_vline(
    xintercept = med_y10,
    linetype = "dashed",
    colour = col_reference,
    linewidth = 0.5,
    alpha = 0.7
  ) +
  annotate(
    "text",
    x = med_y10 + 0.02,
    y = Inf,
    label = sprintf("Median: %.2f%%", med_y10),
    hjust = 0, vjust = 1.5,
    size = 3.4, colour = col_reference
  ) +
  annotate(
    "text",
    x = min(y10$y10_pct), y = Inf,
    label = "Kernel density estimate (red)",
    hjust = 0, vjust = 3.0,
    size = 3.4, colour = col_accent
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.06))) +
  labs(
    x = "10-year zero-coupon rate (%)",
    y = "Number of weekly observations"
  ) +
  theme_article() +
  theme(legend.position = "none")

ggsave("y10_distribution.png", plot = p2,
       width = 9, height = 5, dpi = 140, bg = "white")

# -----------------------------------------------------------------------------
# 5. Summary numbers printed for the article's caption text
# -----------------------------------------------------------------------------

cat("\n--- Sanity check ---\n")
cat(sprintf("Weekly observations: %d\n", nrow(weekly)))
cat(sprintf("10Y rate range: %.3f%% to %.3f%%\n",
            min(y10$y10_pct), max(y10$y10_pct)))
cat(sprintf("10Y mean: %.3f%%, median: %.3f%%\n",
            mean(y10$y10_pct), med_y10))
cat(sprintf("Share below 1.0%%: %.1f%%\n",
            mean(y10$y10_pct < 1) * 100))
cat(sprintf("Share between 1.0%% and 1.5%%: %.1f%%\n",
            mean(y10$y10_pct >= 1 & y10$y10_pct < 1.5) * 100))
cat(sprintf("Share at or above 1.5%%: %.1f%%\n",
            mean(y10$y10_pct >= 1.5) * 100))

cat("\n--- CoV table ---\n")
tenor_stats |>
  mutate(cov = sd_pct / mean_pct) |>
  print(n = 10)

cat("\nFigures written: mean_std_by_tenor.png, y10_distribution.png\n")
