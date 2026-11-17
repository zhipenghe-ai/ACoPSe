"""Train or evaluate on MIRFlickr-25K."""

from configs import build_parser

def param_list(log, config):
    log.info('>>> Configs List <<<')
    log.info('--- Dataset:{}'.format(config.DATASET))
    log.info('--- SEED:{}'.format(config.SEED))
    log.info('--- Bit:{}'.format(config.HASH_BIT))
    log.info('--- Batch:{}'.format(config.BATCH_SIZE))
    log.info('--- Lr_IMG:{}'.format(config.LR_IMG))
    log.info('--- Lr_TXT:{}'.format(config.LR_TXT))
    log.info('--- LR_MyNet:{}'.format(config.LR_MyNet))

    log.info('--- lambda1:{}'.format(config.lambda1))
    log.info('--- lambda2:{}'.format(config.lambda2))
    log.info('--- beta:{}'.format(config.beta))
    log.info('--- K:{}'.format(config.K))
    log.info('--- a1:{}'.format(config.a1))
    log.info('--- a2:{}'.format(config.a2))

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
    else:
        wxz.load_checkpoints()
        wxz.performance_eval()

if __name__ == '__main__':
    main(build_parser('mirflickr').parse_args())
