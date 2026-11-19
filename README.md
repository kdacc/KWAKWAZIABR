## The 2026 NASA Space Apps Challenge 
# Planet X and SPHEREx

**Explore and compare real astronomical observations from NASA's SPHEREx space telescope to investigate how objects change over time.**

## Overview

PlanetKwa is an interactive web application that allows users to explore SPHEREx observations of astronomical regions and compare images captured at different times and spectral channels.

The application provides tools for visualizing observations, identifying potential changes, and documenting findings in downloadable PDF reports.
## Key Features

### 1. Astronomical Search
- Search for astronomical regions using celestial coordinates (RA/Dec).
- Retrieve available SPHEREx observations for the selected coordinates.

### 2. Image Visualization
- Display retrieved SPHEREx observation images.
- Switch between different observation dates.
- Select and explore different spectral channels.
- View basic observation metadata.

### 3. Temporal Image Comparison
- Compare two images captured at different observation times.
- Use **Blink Animation** to switch automatically between observations and identify visual differences.
- Use an interactive **Slider** to compare two images by moving between them.

### 4. Change Inspection
- Inspect regions where potential changes are visible.
- Display coordinates and additional information about identified changes.
- Use observation metadata to support further investigation.

### 5. PDF Report Export
PlanetKwa allows users to document and export their observations as PDF reports.

The report creation process includes:
- Selecting a specific region of an observation where a potential change has been identified.
- Automatically populating technical information, including:
  - Celestial coordinates (RA/Dec).
  - Field of view (FoV).
  - Detector number.
  - Total number of frames.
  - Selected frame number.
- Adding personal observations and descriptions through a report form.
- Generating and downloading the completed report as a PDF file.

This feature helps users preserve their findings and share documented observations for further investigation.

## How It Works

The application follows a straightforward workflow:

1. **Search** — Select an astronomical object from the available list. Its celestial coordinates (RA/Dec) are populated automatically.
2. **Observe** — Browse retrieved SPHEREx images, observation dates, spectral channels, and metadata.
3. **Compare** — Compare observations using image comparison, Blink Animation, or the interactive Slider.
4. **Analyze** — Inspect potential changes and review their coordinates and related information.
5. **Export** — Select a region, add observations, and download a PDF report.

## Data Processing

PlanetKwa processes real SPHEREx observational data to prepare images for visualization and comparison.

The processing pipeline includes retrieving observation data, downloading FITS images, cleaning image data, aligning observations to a common coordinate grid, and preparing image frames for visualization.

## Technology Stack

- **Python** — Core application logic and data processing.
- **NumPy** — Numerical operations and image data processing.
- **Astropy** — Astronomical coordinates, FITS handling, and World Coordinate System (WCS).
- **Astroquery** — Access to astronomical databases and services.
- **Reproject** — Aligning astronomical images to a common coordinate system.
- **Pillow** — Image processing and rendering.
- **NASA SPHEREx / IRSA** — Source of observational data.

## Test Objects

The project includes astronomical targets associated with different types of observable changes:

- Barnard's Star
- Epsilon Indi B
- WISE 0855
- RW Aurigae
- R Draconis

These targets provide examples for exploring phenomena such as proper motion and brightness variability.

## Project Scope

PlanetKwa is an exploratory tool for visualizing and comparing astronomical observations. Differences between images may indicate changes that deserve further investigation, but visual differences alone do not confirm new astronomical discoveries.



**PlanetKwa – Making the changing universe easier to explore.**
