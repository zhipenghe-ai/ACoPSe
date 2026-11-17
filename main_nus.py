"""Train or evaluate on NUS-WIDE."""

from configs import build_parser

def param_list(log, config):
    args_dict = vars(config)

    for key, value in args_dict.items():
        log.info(f'{key} = {value}')

def main(config):
    import torch
    from my_opt import PCIRH, RETRIEVAL_TOP_K
    from utils import logger

    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    torch.cuda.set_device(config.GPU_ID)

    logName = config.DATASET + '_' + str(config.HASH_BIT)
    log = logger(logName, log_dir=getattr(config, 'LOG_DIR', './logs'))
    param_list(log, config)

    wxz = PCIRH(log, config)
    best_it = best_ti = 0

    if config.TRAIN == True:
        wxz.warmup(config.warmup)
        for epoch in range(config.NUM_EPOCH):
            coll_B, record_index = wxz.train_method(epoch)
            wxz.train_Hashfunc(coll_B, record_index, epoch)

            if (epoch + 1) % config.EVAL_INTERVAL == 0:
                MAP_I2T, MAP_T2I = wxz.performance_eval()

                if (best_it + best_ti) < (MAP_I2T + MAP_T2I):
                    best_it, best_ti = MAP_I2T, MAP_T2I
                    log.info('Best MAP@%d I->T: %.3f, MAP@%d T->I: %.3f',
                             RETRIEVAL_TOP_K, best_it, RETRIEVAL_TOP_K, best_ti)
                    wxz.save_checkpoints()

                log.info('--------------------------------------------------------------------')
        if config.EX_EPOCH:
            for epoch in range(config.EX_EPOCH):
                wxz.train_Hashfunc(coll_B, record_index, epoch)
                if (epoch + 1) % 5 == 0:
                    wxz.performance_eval()
    else:
        wxz.load_checkpoints()
        wxz.performance_eval()


if __name__ == '__main__':
    main(build_parser('nuswide').parse_args())
