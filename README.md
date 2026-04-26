# Nebulift
 Nebulift is an adaptive stretching GUI based on max-tree source detection and parameter extraction MTObjects

## Table of Contents
- [Processing Pipeline](#processing-pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Processing Pipeline
The processing pipeline of Nebulift consists of the following steps:
1. **Data Loading**: Load your astronomical data into the GUI.
2. **Max-Tree Source Detection**: Use MTObjects to detect sources in the data using a max-tree structure.
3. **Parameter Extraction**: Extract relevant parameters from the detected sources for further analysis.
4. **Classification**: Classify the detected sources based on their extracted parameters.
5. **Adaptive Stretching**: Apply parameter-based stretching based on the classification results to enhance the visibility of the sources in the data.

## Installation
To install Nebulift, follow these steps:
1. Clone the repository:
```bash
git clone https://github.com/TheAefka/nebulift.git
```
2. Install the required dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
pip install -r requirements.txt
```

## Usage
To run Nebulift, use the following command:
```bash
python nebulift.py
```
This will launch the GUI where you can load your data and perform adaptive stretching using the max-tree source detection and parameter extraction features.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details