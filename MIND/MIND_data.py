import requests
import zipfile
import os
import pandas as pd


def get_data(url:str, destination_folder:str, data_name:str):
    """
    Download relevant data from Figshare page
    :param url: Figshare url such as "https://figshare.com/ndownloader/files/57981334"
    :param destination_folder: the location of the downloaded files
    :param data_name: name of the data
    :return:
    """
    dir = destination_folder
    if not os.path.exists(dir):
        # Create the folder
        os.makedirs(dir)
        print(f"Folder '{dir}' created successfully. ✅")

    local_filename = data_name + '.zip'
    print(f"Downloading from {url}...")

    if not os.path.isfile(dir + local_filename):
        try:
            # 3. Use requests to get the file, stream=True is important for large files.
            with requests.get(url, stream=True) as r:
                r.raise_for_status()  # Raises an exception for bad status codes (4xx or 5xx).

                # 4. Open a local file in binary write mode.
                with open(dir + local_filename, 'wb') as f:
                    # 5. Write the content in chunks.
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            print(f"Successfully downloaded and saved to {dir + local_filename}")

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")

    if not os.path.exists(destination_folder):
        # Create the folder
        os.makedirs(destination_folder)

    try:
        with zipfile.ZipFile(dir + local_filename, 'r') as zip_ref:
            # The extractall() method extracts all contents to the specified folder.
            zip_ref.extractall(destination_folder)

        print(f"Successfully unzipped '{dir + local_filename}' into '{destination_folder}'. ✅")

        os.remove(dir + local_filename)

    except zipfile.BadZipFile:
        print(f"Error: '{dir + local_filename}' is not a valid zip file or is corrupted. ❌")
    except FileNotFoundError:
        print(f"Error: The file '{dir + local_filename}' was not found. ❌")


def get_synthetic(download_dir: str = './MIND_downloaded_data/'):
    """
    Get the preprocessed synthetic data used in the MIND paper
    :param download_dir: User specified directory for downloading preprocessed data
    :return: None
    """
    get_data("https://figshare.com/ndownloader/files/57981334",
             download_dir, 'MIND_synthetic_data')


def load_synthetic(download_dir: str = './MIND_downloaded_data/', mode: str = 'high'):
    """
    Load the synthetic data
    :param download_dir: User specified directory for downloading preprocessed data
    :param mode: 'high' indicates loading the high noise data, 'low' indicates loading the low noise data
    :return: Dictionary of Pandas dataframes, each corresponds to one modality
    """
    sim_3 = pd.read_csv(download_dir + 'synthetic_data/sim_methyl_{}.csv'.format(mode), index_col=0)
    sim_2 = pd.read_csv(download_dir + 'synthetic_data/sim_protein_{}.csv'.format(mode), index_col=0)
    sim_1 = pd.read_csv(download_dir + 'synthetic_data/sim_expr_{}.csv'.format(mode), index_col=0)
    sim_cls = pd.read_csv(download_dir + 'synthetic_data/sim_cls_{}.csv'.format(mode), index_col=1)
    return {'RNA_expr': sim_1, 'Protein': sim_2, 'DNA_methyl': sim_3, 'cls': sim_cls}


def get_TCGA(download_dir: str = './MIND_downloaded_data/'):
    """
    Get the preprocessed TCGA data used in the MIND paper
    :param download_dir: User specified directory for downloading preprocessed data
    :return: None
    """
    get_data("https://figshare.com/ndownloader/files/57981316",
             download_dir, 'MIND_TCGA_data')


def load_TCGA(cancer_type: str, task: str = 'all', download_dir: str = './MIND_downloaded_data/'):
    """
    Load the TCGA data
    :param cancer_type: cancer type of the TCGA data
    :param task: if recon_test, load the masked data, if recon_train, then load the not masked data, if all, load all
    :param download_dir: User specified directory for downloading preprocessed data
    :return: Dictionary of Pandas dataframes, each corresponds to one modality
    """
    mods = ['RNA', 'methyl', 'CNA', 'miRNA', 'RPPA']
    ans = {'clinical': pd.read_csv(download_dir + 'TCGA_preprocessed/{}/clinic_data.csv'.format(cancer_type), header=0, index_col=0)}
    if task == 'recon_test':
        for mod in mods:
            if os.path.isfile(download_dir + 'TCGA_preprocessed/{}/{}_data_test.csv'.format(cancer_type, mod)):
                ans[mod] = pd.read_csv(download_dir + 'TCGA_preprocessed/{}/{}_data_test.csv'.format(cancer_type, mod),
                                       header=0, index_col=0)
        return ans
    elif task == 'recon_train':
        for mod in mods:
            if os.path.isfile(download_dir + 'TCGA_preprocessed/{}/{}_data_train.csv'.format(cancer_type, mod)):
                ans[mod] = pd.read_csv(download_dir + 'TCGA_preprocessed/{}/{}_data_train.csv'.format(cancer_type, mod), header=0,
                                         index_col=0)
        return ans
    else:
        ans['RNA'] = pd.read_csv(download_dir + 'TCGA_preprocessed/{}/RNA_data.csv'.format(cancer_type), header=0, index_col=0)
        ans['methyl'] = pd.read_csv(download_dir + 'TCGA_preprocessed/{}/meth_data.csv'.format(cancer_type), header=0, index_col=0)
        ans['RPPA'] = pd.read_csv(download_dir + 'TCGA_preprocessed/{}/rppa_data_imp.csv'.format(cancer_type), header=0, index_col=0)
        ans['CNA'] = pd.read_csv(download_dir + 'TCGA_preprocessed/{}/cna_data.csv'.format(cancer_type), header=0, index_col=0)
        if os.path.isfile(download_dir + 'TCGA_preprocessed/{}/miRNA_data_imp.csv'.format(cancer_type)):
            ans['miRNA'] = pd.read_csv(download_dir + 'TCGA_preprocessed/{}/miRNA_data_imp.csv'.format(cancer_type), header=0,
                                     index_col=0)
        return ans


def get_CCMA(download_dir: str = './MIND_downloaded_data/'):
    """
    Get the preprocessed CCMA data used in the MIND paper
    :param download_dir: User specified directory for downloading preprocessed data
    :return: None
    """

    get_data("https://figshare.com/ndownloader/files/57981304",
             download_dir, 'MIND_CCMA_data')


def load_CCMA(task: str = 'all', download_dir: str = './MIND_downloaded_data/'):
    """
    Load the CCMA data
    :param task: if recon_test, load the masked data, if recon_train, then load the not masked data, if all, load all
    :param download_dir: User specified directory for downloading preprocessed data
    :return: Dictionary of Pandas dataframes, each corresponds to one modality
    """
    ans = {'clinical': pd.read_csv(download_dir + 'CCMA_preprocessed/clinical.csv', header=0, index_col=0)}
    if task == 'all':
        ans['RNA'] = pd.read_csv(download_dir + 'CCMA_preprocessed/mRNA.csv', header=0, index_col=0)
        ans['methyl'] = pd.read_csv(download_dir + 'CCMA_preprocessed/meth.csv', header=0, index_col=0)
        ans['CNV'] = pd.read_csv(download_dir + 'CCMA_preprocessed/CNV.csv', header=0, index_col=0)
    elif task == 'recon_train':
        ans['RNA'] = pd.read_csv(download_dir + 'CCMA_preprocessed/mRNA_train.csv', header=0, index_col=0)
        ans['methyl'] = pd.read_csv(download_dir + 'CCMA_preprocessed/meth_train.csv', header=0, index_col=0)
        ans['CNV'] = pd.read_csv(download_dir + 'CCMA_preprocessed/CNV_train.csv', header=0, index_col=0)
    else:
        ans['RNA'] = pd.read_csv(download_dir + 'CCMA_preprocessed/mRNA_test.csv', header=0, index_col=0)
        ans['methyl'] = pd.read_csv(download_dir + 'CCMA_preprocessed/meth_test.csv', header=0, index_col=0)
        ans['CNV'] = pd.read_csv(download_dir + 'CCMA_preprocessed/CNV_test.csv', header=0, index_col=0)

    return ans


def get_CCLE(download_dir: str = './MIND_downloaded_data/'):
    """
    Get the preprocessed CCLE data used in the MIND paper
    :param download_dir: User specified directory for downloading preprocessed data
    :return: None
    """
    get_data("https://figshare.com/ndownloader/files/57981310",
             download_dir, 'MIND_CCLE_data')


def load_CCLE(task: str = 'all', download_dir: str = './MIND_downloaded_data/'):
    """
    Load the CCLE data
    :param task: if recon_test, load the masked data, if recon_train, then load the not masked data, if all, load all
    :param download_dir: User specified directory for downloading preprocessed data
    :return: Dictionary of Pandas dataframes, each corresponds to one modality
    """
    ans = {'clinical': pd.read_csv(download_dir + 'CCLE_preprocessed/clinic_data.csv', header=0, index_col=0)}
    if task == 'all':
        ans['RNA'] = pd.read_csv(download_dir + 'CCLE_preprocessed/RNA_data.csv', header=0, index_col=0)
        ans['meth'] = pd.read_csv(download_dir + 'CCLE_preprocessed/meth_data.csv', header=0, index_col=0)
        ans['cna'] = pd.read_csv(download_dir + 'CCLE_preprocessed/cna_data.csv', header=0, index_col=0)
        ans['metabolomics'] = pd.read_csv(download_dir + 'CCLE_preprocessed/metabolomics_data.csv', header=0, index_col=0)
        ans['miRNA'] = pd.read_csv(download_dir + 'CCLE_preprocessed/miRNA_data.csv', header=0, index_col=0)
        ans['rppa'] = pd.read_csv(download_dir + 'CCLE_preprocessed/rppa_data.csv', header=0, index_col=0)
    elif task == 'recon_train':
        ans['RNA'] = pd.read_csv(download_dir + 'CCLE_preprocessed/RNA_data_train.csv', header=0, index_col=0)
        ans['meth'] = pd.read_csv(download_dir + 'CCLE_preprocessed/meth_data_train.csv', header=0, index_col=0)
        ans['cna'] = pd.read_csv(download_dir + 'CCLE_preprocessed/cna_data_train.csv', header=0, index_col=0)
        ans['metabolomics'] = pd.read_csv(download_dir + 'CCLE_preprocessed/metabolomics_data_train.csv', header=0, index_col=0)
        ans['miRNA'] = pd.read_csv(download_dir + 'CCLE_preprocessed/miRNA_data_train.csv', header=0, index_col=0)
        ans['rppa'] = pd.read_csv(download_dir + 'CCLE_preprocessed/rppa_data_train.csv', header=0, index_col=0)
    else:
        ans['RNA'] = pd.read_csv(download_dir + 'CCLE_preprocessed/RNA_data_test.csv', header=0, index_col=0)
        ans['meth'] = pd.read_csv(download_dir + 'CCLE_preprocessed/meth_data_test.csv', header=0, index_col=0)
        ans['cna'] = pd.read_csv(download_dir + 'CCLE_preprocessed/cna_data_test.csv', header=0, index_col=0)
        ans['metabolomics'] = pd.read_csv(download_dir + 'CCLE_preprocessed/metabolomics_data_test.csv', header=0, index_col=0)
        ans['miRNA'] = pd.read_csv(download_dir + 'CCLE_preprocessed/miRNA_data_test.csv', header=0, index_col=0)
        ans['rppa'] = pd.read_csv(download_dir + 'CCLE_preprocessed/rppa_data_test.csv', header=0, index_col=0)
    return ans




