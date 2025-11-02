"""
Configuration module for Weda Bay Watch remote sensing pipeline.
"""
from dataclasses import dataclass, field
from typing import Dict, List
import json
from pathlib import Path
import os


@dataclass
class BaseClassifier:
    class_int_mapping: Dict[str, int] = field(default_factory=lambda: {
        'No data': -1,
        'Cloud': 0,
        'Jungle': 1,
        'Cleared': 2,
        'Ocean': 3
    })

    int_class_mapping: Dict[int, str] = field(default_factory=lambda: {
        -1: 'No data',
        0: 'Cloud',
        1: 'Jungle',
        2: 'Cleared',
        3: 'Ocean'
    })

    colours: Dict[str, str] = field(default_factory=lambda: {
        "No data": "black",
        "Cloud": "lightgrey",
        "Jungle": "forestgreen",
        "Cleared": "orange",
        "Ocean": "blue"
    })

@dataclass
class BandConfig:
    """Band configuration for different Landsat sensors."""
    LE07: List[str] = field(default_factory=lambda: ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'QA_PIXEL'])
    LC08: List[str] = field(default_factory=lambda: ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'QA_PIXEL'])
    LC09: List[str] = field(default_factory=lambda: ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'QA_PIXEL'])

    def to_dict(self):
        return {
            'LE07': self.LE07,
            'LC08': self.LC08,
            'LC09': self.LC09
        }
        
@dataclass
class BandMap:
    """Spectral band mappings for different Landsat sensors."""
    LC09: Dict[str, str] = field(default_factory=lambda: {
        'AEROSOL': 'B1',
        'B': 'B2',
        'G': 'B3',
        'R': 'B4',
        'NIR': 'B5',
        'SWIR1': 'B6',
        'SWIR2': 'B7',
        'TIR1': 'B10',
        'TIR2': 'B11'
    })
    LC08: Dict[str, str] = field(default_factory=lambda: {
        'AEROSOL': 'B1',
        'B': 'B2',
        'G': 'B3',
        'R': 'B4',
        'NIR': 'B5',
        'SWIR1': 'B6',
        'SWIR2': 'B7',
        'TIR1': 'B10',
        'TIR2': 'B11'
    })
    LE07: Dict[str, str] = field(default_factory=lambda: {
        'B': 'B1',
        'G': 'B2',
        'R': 'B3',
        'NIR': 'B4',
        'SWIR1': 'B5',
        'TIR': 'B6',
        'SWIR2': 'B7'
    })

    def to_dict(self):
        return {
            'LC09': self.LC09,
            'LC08': self.LC08,
            'LE07': self.LE07
        }


@dataclass
class CompositeConfig:
    """Configuration for spectral indices and composites."""
    enabled: List[str] = field(default_factory=lambda: [
        'RGB', 'NDVI', 'MNDWI', 'NDBI', 'EVI', 'SAVI', 'FERRIC_IRON', 'BAI', 'SI', 'NDGI', 'NDMI', 'SR', 'BSI', 'FeO'
    ]) # CMI (same as NDBI)
    
    # Define which composites to use for specific analyses
    vegetation_indices: List[str] = field(default_factory=lambda: ['NDVI', 'EVI', 'SAVI', 'NDGI'])
    water_indices: List[str] = field(default_factory=lambda: ['MNDWI'])
    urban_indices: List[str] = field(default_factory=lambda: ['NDBI'])
    
    def get_all(self):
        return self.enabled


@dataclass
class ROIConfig:
    """Configuration for Regions of Interest."""
    all_rois: List[str] = field(default_factory=lambda: [
        'close_forest_1', 
        'close_forest_2', 
        'untouched_forest_1', 
        'untouched_forest_2', 
        'untouched_forest_3',
        'greater_mine_site',
        'ocean_near_mine_site',
        'river_outlet',
        'water_body'
    ])

    colors: Dict[str, str] = field(default_factory=lambda: {
        'close_forest_1': '#7FBF00',
        'close_forest_2': '#99CC00',
        'greater_mine_site': '#A04020',
        'ocean_near_mine_site': '#1f77b4',
        'river_outlet': '#5a7b8b',
        'untouched_forest_1': '#006400',
        'untouched_forest_2': '#004d00',
        'untouched_forest_3': '#003300',
        'water_body': '#008080',
        'burnt_forest': '#B8860B'
    })
    
    # ROI groups for analysis
    forest_rois: List[str] = field(default_factory=lambda: [
        'close_forest_1', 'close_forest_2', 
        'untouched_forest_1', 'untouched_forest_2', 'untouched_forest_3'
    ])
    
    mine_rois: List[str] = field(default_factory=lambda: [
        'greater_mine_site'
    ])
    
    water_rois: List[str] = field(default_factory=lambda: [
        'ocean_near_mine_site', 'river_outlet', 'water_body'
    ])


@dataclass
class PathConfig:
    """Configuration for project paths."""
    root: str = '.'
    data: str = 'data'
    raw: str = 'data/raw'
    clipped: str = 'data/clipped'
    stacked: str = 'data/stacked'
    derived: str = 'data/derived'
    classified: str = 'output/classified'
    initial_analysis: str = 'output/initial_analysis'
    pixel_regression: str = 'output/pixel_regression'
    luc_analysis: str = 'output/luc_analysis'
    grid_analysis: str = 'output/grid_analysis'
    proximity_analysis: str = 'output/proximity_analysis'  # New: for proximity analysis
    rois: str = 'output/rois'
    rois_initial: str = 'output/rois/initial_analysis'
    rois_classification: str = 'output/rois/classification_training_rois'
    tests: str = 'tests'
    envi: str = 'data/envi'

    smoothed_stack_all_timestamps_file_path: str = 'output/classified/smoothed_stack.npy'
    jungle_to_mine_change_tif_file_path: str = 'output/classified/jungle_to_mine_change.tif'
    initial_analysis_all_stats_pkl_file_path: str = os.path.join(initial_analysis, 'all_stats.pkl')
    example_rgb_image_1: str = os.path.join(derived, 'LC08_L2SP_109060_20230506_20230509_02_T1/RGB')
    example_rgb_image_2: str = os.path.join(derived, 'LC08_L2SP_109060_20160603_20200907_02_T1/RGB')

    last_image: str = os.path.join(derived, 'LC09_L2SP_109060_20250722_20250725_02_T1')
    
    def get_path(self, key: str) -> Path:
        """Get Path object for a given key."""
        return Path(getattr(self, key))


@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters."""
    # Grid analysis parameters
    grid_size_meters: int = 500  # Size of each grid cell in meters
    min_valid_pixels_pct: float = 0.7  # Minimum % of non-cloud pixels required
    
    # Proximity analysis parameters
    buffer_distances_meters: List[int] = field(default_factory=lambda: [500, 1000, 2000, 5000, 10000])
    
    # Cloud masking
    max_cloud_fraction: float = 0.3  # Maximum cloud cover to include in analysis
    
    # Temporal analysis
    start_year: int = 2016
    end_year: int = 2025


@dataclass
class ProjectConfig:
    """Master configuration for Weda Bay Watch project."""
    bands: BandConfig = field(default_factory=BandConfig)
    band_map: BandMap = field(default_factory=BandMap)
    composites: CompositeConfig = field(default_factory=CompositeConfig)
    rois: ROIConfig = field(default_factory=ROIConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    classifier: BaseClassifier = field(default_factory=BaseClassifier)
    
    def save_to_json(self, filepath: str):
        """Save configuration to JSON file."""
        config_dict = {
            'bands': self.bands.to_dict(),
            'composites': {
                'enabled': self.composites.enabled,
                'vegetation_indices': self.composites.vegetation_indices,
                'water_indices': self.composites.water_indices,
                'urban_indices': self.composites.urban_indices
            },
            'rois': {
                'colors': self.rois.colors,
                'forest_rois': self.rois.forest_rois,
                'mine_rois': self.rois.mine_rois,
                'water_rois': self.rois.water_rois
            },
            'paths': self.paths.__dict__,
            'analysis': {
                'grid_size_meters': self.analysis.grid_size_meters,
                'min_valid_pixels_pct': self.analysis.min_valid_pixels_pct,
                'buffer_distances_meters': self.analysis.buffer_distances_meters,
                'max_cloud_fraction': self.analysis.max_cloud_fraction,
                'start_year': self.analysis.start_year,
                'end_year': self.analysis.end_year
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_from_json(cls, filepath: str):
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct config objects
        bands = BandConfig(**data['bands'])
        composites = CompositeConfig(**data['composites'])
        rois = ROIConfig(**data['rois'])
        paths = PathConfig(**data['paths'])
        analysis = AnalysisConfig(**data['analysis'])
        
        return cls(bands=bands, composites=composites, rois=rois, paths=paths, analysis=analysis)


# Create default config instance
config = ProjectConfig()


# Convenience function to get config
def get_config() -> ProjectConfig:
    """Get the project configuration."""
    return config


# if __name__ == '__main__':
#     # Example: Save default config
#     config.save_to_json('config.json')
#     print("Default configuration saved to config.json")
    
#     # Example: Load config
#     loaded_config = ProjectConfig.load_from_json('config.json')
#     print(f"Loaded config with {len(loaded_config.composites.enabled)} composites")