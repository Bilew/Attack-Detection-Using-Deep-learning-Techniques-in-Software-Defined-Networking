# Attack-Detection Using Deep-learning in Software-Defined Networking  
# Overview
This project aims to detect cyber attacks using deep learning techniques in SDN. By leveraging advanced Deep learning models, the system can identify potential threats in network traffic and security logs.
Software-Defined Networking (SDN) is an innovative approach to networking that separates the control plane from the data plane, allowing for more flexible and programmable network management. SDN enables network administrators to control and configure the network through software applications, rather than relying on traditional hardware-based configurations. This offers greater scalability, easier management, and the ability to dynamically adjust the network to changing demands.


## Features
- **Deep Learning Models**: Uses CNNs, RNNs, LSTMs, or transformer-based architectures for attack detection.
- **Real-Time Detection**: Provides near-instantaneous detection of threats.
- **Dataset Handling**: Supports multiple cybersecurity datasets, such as CIC-IDS2017, NSL-KDD, and UNSW-NB15.
- **Scalability**: Can be integrated into security information and event management (SIEM) systems.
- **Visualization & Reporting**: Offers logs and visual dashboards for analysis.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python (>=3.5)
- TensorFlow 
- Scikit-learn
- Pandas
- Matplotlib / Seaborn
- Jupyter Notebook (optional, for experiments)

### Setup
Clone the repository and install dependencies:
```sh
git clone https://github.com/bilew/Attack-Detection-Using-Deep-learning.git
cd Attack-Detection Using Deep-learning in Software-Defined Networking 
pip install -r requirements.txt
```

## Usage
1. **Prepare Dataset**
   - Download and preprocess the dataset.
   - the data is Split into training and testing sets.

2. **Train Model**
   python train.py --dataset dataset_name --model model_type
3. **Evaluate Model**
    - check the model evaluation 
## Dataset Sources
- [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html)
## Model Architectures
- LSTMs**: Effective for sequential data analysis.

## Prepare a network Diagram using:  
  - Mininet
  - Ryu Controller
    
## From the Ryu Contrller in the control Layer: 
   - On top of your topology diagram, test the performance of deep learning for detecting attacks/malware.
   - Then, evaluate the performance.
## Results & Performance
- Accuracy: **97.55**
- Precision: **96.48%**
- Recall: **99.01**
- F1-Score: **97.72**


## Contributing
Feel free to contribute to the project by submitting pull requests. Please ensure that your contributions align with best practices in deep learning and cybersecurity.

## License
None.

## Contact
For questions or collaborations, reach out at **bilew19@gmail.com** or open an issue in the repository.

# Deep Learning Models:
Uses RNNs, LSTMs, or transformer-based architectures for attack detection.
Real-Time Detection: Provides near-instantaneous detection of threats.
- for future 
Dataset Handling: Supports multiple cybersecurity datasets, such as CIC-IDS2017, NSL-KDD, and UNSW-NB15.
Scalability: Can be integrated into security information and event management (SIEM) systems.
Visualization & Reporting: Offers logs and visual dashboards for analysis.

# For Future:
I have a dream to providing real-time malware detection in Software-Deffined Networking using advanced Deep Learning Techniques like Bidirectional LSTM.

