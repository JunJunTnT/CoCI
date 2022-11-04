import imp
import os.path as osp


def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    print(pathname)
    print("---")
    print(type(imp.load_source('', pathname)))
    return imp.load_source('', pathname)
