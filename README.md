# A Data-Driven Study on the Hawthorne Effect in Sensor-Based Human Activity Recognition

## Abstract
Known as the Hawthorne Effect, studies have shown that participants alter their behavior and execution of activities in response to being observed. With researchers from a multitude of humancentered studies knowing of the existence of the said effect, quantitative studies investigating the neutrality and quality of data gathered in monitored versus unmonitored setups, particularly in the context of Human Activity Recognition (HAR), remain largely underexplored. With the development of tracking devices providing the possibility of carrying out less invasive observation of participants’ conduct, this study provides a data-driven approach of measuring the effects of observation on participants’ execution of five workout-based activities. Using both classical feature analysis and deep learning-based methods we analyze the accelerometer data of 10 participants, showing that a different degree of observation only marginally influences captured patterns and predictive performance of classification algorithms. Although our findings do not dismiss the existence of the Hawthorne Effect, it does challenge the prevailing notion of the applicability of laboratory compared to in-the-wild recorded data.

## Changelog
- 03/07/2023: initial commit.

## Installation
Please follow instructions mentioned in the [INSTALL.md](/INSTALL.md) file.

## Dataset
The full dataset can be found in the folder [data](/data/). The subfolders represent the data split into ways, used for the three different type of experiments:
- ``day``: dataset split into files representing one workout of one subject on a specific day. The day corresponds to the type of session mentioned in the main paper, i.e. ``day_0`` is ``session_0``.
- ``loso``: dataset split into files representing all workouts of one subject. Labels are the performed activities (+ NULL-class)
- ``session``: dataset split into files representing all workouts of one subject. Data records are labeled using the session identifier, i.e. there being 4 labels (``session_0``, ``session_1``, ``session_2`` and ``session_3``).

## Reproduce Experiments
Once having installed requirements, one can rerun experiments by running the `main.py` script:

````
python main.py --config ./configs/deepconvlstm_session.yaml --seed 1 --eval_type split
````

Each config file represents one type of experiment. Each experiment was run three times using three different random seeds (i.e. `1, 2, 3`).

### Logging using Neptune.ai

In order to log experiments to [Neptune.ai](https://neptune.ai) please provide `project`and `api_token` information in your local deployment (see lines `32-33` in `main.py`)

## Contact
Marius Bock (marius.bock@uni-siegen.de)
