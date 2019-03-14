Prerequisites:
python 3.6 with libs such as uiautomator, subprocess, numpy, scikit-learn, bs4, etc.
Dynamic taint analysis supported firmware, such as TaintDroid.


AppInspector

0. Set up a clean TaintDroid environment, with UiDroid_TaintNotify installed (to extract logs from TaintDroid). 
1. Run the exerciser (e.g. UIDroid) to automatically collect sensitive transmissions and the corresponding app-level contexts. 
2. Run pcap_tdroid_matcher to filter app contexts (with the pcaps) who do not generate any sensitive flows.
3. (cd the filtered directory, manually label the app contexts to be 'expected' or not based on the sensitive info transmitted.)
4. Having the labeled contexts, run "ContextProcessor.py <data dir>" to build ML models.

TrafficAnalyzer

1. analyzer.py to analyze the traffic data specified by illegal contexts, or any self-provided flows.
2. predictor.py to leverage the model trained by analyzer to predict unseen data.  

If you are looking for the conference paper, please click [here](https://ieeexplore.ieee.org/abstract/document/7732993/).

The bibtex:
```
@inproceedings{flowintent,
  title={FlowIntent: Detecting Privacy Leakage from User Intention to Network Traffic Mapping},
  author={Fu, Hao and Zheng, Zizhan and Das, Aveek K and Pathak, Parth H and Hu, Pengfei and Mohapatra, Prasant},
  booktitle={IEEE International Conference on Sensing, Communication, and Networking (SECON)},
  year={2016}
}
```
The dataset is available [here](https://drive.google.com/file/d/1RVbcSNenhKz_eDPZRv97Q6Ay1FsSvURD/view?usp=sharing).