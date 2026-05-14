## CarRacing CNN Agent

Reinforcement Learning project using PyTorch and Gymnasium.
The agent learns autonomous driving using CNN and Cross-Entropy Method.

## Technologies
- Python
- PyTorch
- Gymnasium
- OpenCV
- NumPy

## Run
pip install -r requirements.txt
python main.py

## Results

The agent achieved stable results around 100 reward points after approximately 80 training iterations, 
indicating successful learning of stable driving behavior.

### Reward interpretation
- 0 → Stable driving; the agent completes the track without leaving the road or spinning.
- > 0 → Good driving performance.
- < 0 → Unstable or random behavior.
