# WR-One2Set

Our implementation is built on the source code from [One2Set](https://github.com/jiacheng-ye/kg_one2set), [keyphrase-generation-rl](https://github.com/kenchan0226/keyphrase-generation-rl) and [fastNLP](https://github.com/fastnlp/fastNLP).
Thanks for their work.


## Dependency

- python 3.5+
- pytorch 1.0+

## Dataset

The datasets can be downloaded from [here](https://drive.google.com/file/d/16d8nxDnNbRPAw2pVy42DjSTVnT0WzJKj/view?usp=sharing)

## Quick Start

- 1) Finetune the pretrained One2Set-based model provided by the authers of One2Set, which can be downloaded from [here](https://drive.google.com/file/d/184DEgiIkQqJubIxiYiXepZnhhDuNoD5l/view?usp=sharing).
    ```bash
    bash scripts/run_wr_one2set_with_pretrained.sh
    ```

- 2) Or directly train a WR-One2Set-based model.
    ```bash
    bash scripts/run_wr_one2set.sh
    ```

here, we use the seeds 27, 527, 9527 to train our model, which is the same as the previous studies ([One2Set](https://github.com/jiacheng-ye/kg_one2set), [keyphrase-generation-rl](https://github.com/kenchan0226/keyphrase-generation-rl), etc.).






