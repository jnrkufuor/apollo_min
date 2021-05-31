import os
import pandas as pd


class Util(object):

    def __init__(self):
        pass

    def print_to_csv(self, data, outname, outdir="data"):
        ''' Function to print news variable to csv file

            :param news: dataframe containing news data
            :param outname: name of file to be saved
            :param outdir: target directory(will be created if does not exist)
        '''
        if os.name == "nt":
            outdir = ".\\"+outdir
        if os.name == "posix":
            outdir = "./"+outdir

        if not os.path.exists(outdir):
            os.mkdir(outdir)

        fullname = os.path.join(outdir, outname)
        print("Saving to"+fullname)
        data.to_csv(fullname)
        print("done")

    def save_figure(self, g, outname, outdir="images"):
        ''' Function to save a figure

            :g: Seaborn object
            :param outname: name of file to be saved
            :param outdir: target directory(will be created if does not exist)
        '''
        if os.name == "nt":
            outdir = ".\\"+outdir
        if os.name == "posix":
            outdir = "./"+outdir

        if not os.path.exists(outdir):
            os.mkdir(outdir)

        fullname = os.path.join(outdir, outname)
        print("Saving to"+fullname)
        g.get_figure().savefig(fullname, dpi=1000)
        print("done")
    


if __name__ == "__main__":

    ut = Util()
    data = pd.DataFrame(["milk","milk"])
    ut.print_to_csv(data,"test.csv")
