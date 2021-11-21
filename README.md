# Neural Chess

A neural chess bot, written in Jax. Uses a simple transformer-based classifier to predict the next move, conditional
on the player's ELO rating.

Still a WIP. You can play against the current version here:
https://lichess.org/@/transformer_chess

    

TODO:
- [x] Design the data representation and model
- [x] Basic data pipeline and loader
- [x] Supervised policy network, trained on Lichess data (WIP)
- [ ] Self-play / reinforcement learning approaches
