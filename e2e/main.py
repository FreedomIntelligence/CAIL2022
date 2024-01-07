from extractor import extract, preprocess
from converter import create_test_dataset
from extracter_util.postprocess import postprocess1
from generator import generate
from generator_util.postprocess import postprocess2
if __name__ == '__main__':
    create_test_dataset()
    preprocess()
    extract(ensemble=True)
    postprocess1()
    generate()
    postprocess2()

