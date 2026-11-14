## The 2026 NASA Space Apps Challenge 
# Planet X and SPHEREx

PlanetKwa is an interactive web application that allows users to explore, visualize, and compare SPHEREx infrared observations of astronomical objects across different observation dates. It brings astronomical data retrieval, image processing, temporal comparison, and change analysis together in one interface.

## Overview

Repeated observations of the same region of the sky can reveal changes in astronomical objects, including variations in brightness and apparent position. PlanetKwa makes these observations easier to explore by providing tools for retrieving and comparing images from different points in time.

Users can search for an astronomical object, browse its available observations, switch between observation dates, and visualize differences between images.

## Key Features

### Astronomical Object Search
- Search for astronomical objects by name.
- Search by celestial coordinates (Right Ascension and Declination).
- Retrieve available SPHEREx observations for the selected sky region.
- Browse observations from different dates.

### Image Visualization
- View processed SPHEREx observation images.
- Switch between available observation dates.
- Explore observations from different SPHEREx spectral channels.
- Access observation metadata, including dates and coordinates.

### Temporal Image Comparison

PlanetKwa provides several ways to compare astronomical observations:

- **Side-by-Side Comparison** — display two observations next to each other.
- **Slider Comparison** — interactively move between two images to inspect differences.
- **Blink Animation** — rapidly alternate between observations to make changes easier to notice.
- **Difference Map** — visualize pixel-level differences between observations to highlight potential changes.

### Change Analysis
- Highlight potential brightness variations and positional changes.
- Display relevant observation dates and coordinates.
- Provide information about detected differences to support further investigation.

## How It Works

PlanetKwa follows a straightforward workflow:

1. **Search** – enter an astronomical object's name or RA/Dec coordinates.
2. **Retrieve** – find available SPHEREx observations covering the selected sky region.
3. **Visualize** – load and display observation images with their associated metadata.
4. **Compare** – select observations from different dates and use the available comparison modes.
5. **Analyze** – inspect visual differences and explore potential changes in the astronomical object.

## Data Processing

The application processes real SPHEREx observational data to make images suitable for temporal comparison.

The processing pipeline includes:

- Querying the SPHEREx archive through the IRSA TAP service.
- Retrieving FITS cutouts for the selected sky region.
- Removing zodiacal light contribution and handling flagged pixels.
- Aligning observations to a common celestial coordinate grid using World Coordinate System (WCS).
- Normalizing image data for consistent visualization.
- Generating image frames and difference maps.
- Caching processed data to reduce repeated downloads and processing time.

## Test Objects

The application supports exploration of astronomical objects selected to demonstrate different types of temporal behavior:

- **Barnard's Star**
- **Epsilon Indi B**
- **WISE 0855**
- **RW Aurigae**
- **R Draconis**

These objects provide examples of phenomena such as proper motion and brightness variability.

## Technology Stack

**Core data processing and astronomy:**
- Python
- NumPy
- Astropy
- Astroquery
- Reproject
- FITS
- Pillow

**Astronomical data services:**
- NASA SPHEREx
- IRSA (Infrared Processing and Analysis Center)

## Project Scope

PlanetKwa focuses on making real SPHEREx observations accessible through an interactive visualization and comparison workflow.

The application is designed for exploring potential changes in astronomical observations. Visual differences are intended to support further investigation and should not be interpreted as automatic confirmation of new planets or other astronomical discoveries.

---

**PlanetKwa — Making the changing universe easier to explore.**
