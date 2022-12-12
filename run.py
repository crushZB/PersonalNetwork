from util.common_utils import parse_yaml
from train import teacher_train
from model.architecture.mymodel import Teacher

if __name__ == '__main__':
    args = parse_yaml('./config.yaml')
    model = Teacher().cuda()
    teacher_train(model, args)
