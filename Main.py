from DnnAdapter import DnnAdapter
import argparse
import sys


def main():
    ap = argparse.ArgumentParser(description="Console interface for various machine-learning models")
    ap.add_argument("-p", "--prototxt", required=True, help="path to 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True, help="path to pre-trained model")
    ap.add_argument("-t", "--task", required=True, help="task type")
    ap.add_argument("-i", "--image", required=True, help="path to image")
    #
    ap.add_argument("-n", "--net", required=False)
    #
    args = vars(ap.parse_args())
    model = DnnAdapter(args)
    model.process_image(args["image"])


if __name__ == '__main__':
    sys.exit(main())
