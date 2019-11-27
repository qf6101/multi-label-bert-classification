import os
import shutil


def pickup_model_file_(args, tag, certain_step):
    dest = args.model_file + tag

    if os.path.exists(dest):
        os.remove(dest)

    if certain_step:
        model_file = os.path.join(args.model_dir, 'model.ckpt-' + certain_step) + tag
        shutil.copyfile(os.path.join(args.model_dir, model_file), dest)
    else:
        with open(os.path.join(args.model_dir, 'checkpoint'), 'r') as f:
            line = f.readlines()[-1]
            model_file = line.split(':')[1].strip().strip('\"') + tag
            shutil.copyfile(os.path.join(args.model_dir, model_file), dest)

    print('Use model file: {}'.format(model_file))


def pickup_model_file(args, certain_step=None):
    pickup_model_file_(args, '.data-00000-of-00001', certain_step)
    pickup_model_file_(args, '.meta', certain_step)
    pickup_model_file_(args, '.index', certain_step)
