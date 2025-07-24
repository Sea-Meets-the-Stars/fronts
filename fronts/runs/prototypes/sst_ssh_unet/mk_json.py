
from fronts.utils import io as utils_io

def main():
    extract_dict = {'fields': ['SST','SSS','Divb2'],
            'field_size': 64,
            'pdicts': 
                {
                    'SST': 
                    {
                    'fixed_km': 144.,
                    'field_size': 64,
                    "quality_thresh": 2,
                    "nrepeat": 1,
                    "downscale": False,
                    "inpaint": False,
                    "median": False,
                    "only_inpaint": False
                    }
                ,
                    'SSS': 
                    {
                    'fixed_km': 144.,
                    'field_size': 64,
                    'smooth_km': 40., 
                    "quality_thresh": 2,
                    "nrepeat": 1,
                    "downscale": False,
                    "inpaint": False,
                    "de_mean": False,
                    "median": False,
                    "only_inpaint": False
                    }
                ,
                    'Divb2': 
                    {
                    'fixed_km': 144.,
                    'field_size': 64,
                    'dx': 144./64,
                    }
                ,
                }
            }
    json_file = 'llc4320_sst144_sss40_extract.json'
    utils_io.savejson('llc4320_sst144_sss40_extract.json', 
                      extract_dict, overwrite=True, easy_to_read=True)
    print(f'Saved {json_file}')

# Command line execution
if __name__ == '__main__':

    main()