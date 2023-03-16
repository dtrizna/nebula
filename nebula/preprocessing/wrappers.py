import os
import time
import logging
import numpy as np
import orjson, json
from lxml import etree
from tqdm import tqdm

from ..models.neurlux import NeurLuxPreprocessor
from ..models.quovadis import QuoVadisModel, report_to_apiseq
from . import JSONTokenizerBPE, PEDynamicFeatureExtractor, JSONTokenizerWhiteSpace
from .normalization import read_and_filter_json_folders

def parse_cruparamer_xmlsample(xml_file):
    # reading from file
    parser = etree.XMLParser(recover=True)
    with open(xml_file) as fobj:
        xml = fobj.read()
    xmlsample = etree.fromstring(xml, parser=parser)

    # parsing structure
    parsed_with_args = []
    parsed_api_only = []
    for action in xmlsample.xpath("//report//file_list//file//start_boot//action_list")[0].getchildren():
        name = action.get("api_name")
        if name:
            name = name.lower()
            args, exinfo = action.getchildren()
            args = " ".join([x.values()[0].lower() for x in args.getchildren()])
            exinfo = " ".join([x.values()[0].lower() for x in exinfo.getchildren()])
            parsed_with_args.append(f"{name} {args} {exinfo}".strip())
            parsed_api_only.append(name)
        #else:
            #logging.error(f"[!] Got action w/o 'api_name': {action.items()}")
            # All actions are the following:
            # <action ID="67" call_name="" call_pid="0" call_time="0" type="System"><exInfo_list count="0" />
            # <description value="Turn off the power." />
    return parsed_with_args, parsed_api_only


def preprocess_neurlux(
        report_paths,
        y=None,
        outfolder=None,
        vocab_size=50000,
        seq_len=512,
        vocab_file=None,
        limit=None,
        in_memory=False,
):
    suffix = "test" if vocab_file else "train"
    limit = limit if limit else "full"
    if outfolder is None:
        outfolder = f"neurlux_{int(time.time())}"
    if not os.path.exists(outfolder):
        os.makedirs(outfolder, exist_ok=True)
    
    if os.path.exists(os.path.join(outfolder, f"x_{suffix}_{limit}.npy")):
        logging.warning(f" [!] Skipping since exists: {os.path.join(outfolder, f'x_{suffix}_{limit}.npy')}")
        return None, os.path.join(outfolder, f"vocab_{vocab_size}.json")

    if in_memory:
        reports_raw = report_paths
    else:
        logging.warning(" [*] Reading raw records...")
        reports_raw = []
        for file in tqdm(report_paths):
            with open(file) as f:
                reports_raw.append(orjson.loads(f.read()))

    neurlux_preprocessor = NeurLuxPreprocessor(
        vocab_size=vocab_size,
        max_length=seq_len
    )
    if vocab_file:
        logging.warning(f" [*] Loading NeurLux preprocessor from {vocab_file}...")
        neurlux_preprocessor.load_vocab(vocab_file)
    else:
        logging.warning(f" [*] Training NeurLux preprocessor on {len(reports_raw)} reports...")
        neurlux_preprocessor.train(reports_raw)
        vocab_file = neurlux_preprocessor.dump_vocab(outfolder)
    
    logging.warning(" [*] Encoding and padding reports...")
    x = neurlux_preprocessor.preprocess_sequence(reports_raw)
    x = neurlux_preprocessor.pad_sequence(x)
    np.save(os.path.join(outfolder, f"x_{suffix}_{limit}.npy"), x)
    logging.warning(f" [!] Saved X as {os.path.join(outfolder, f'x_{suffix}_{limit}.npy')}")
    if y is not None:
        assert len(reports_raw) == y.shape[0], "X and Y must have the same length"
        np.save(os.path.join(outfolder, f"y_{suffix}_{limit}.npy"), y)
        logging.warning(f" [!] Saved Y as {os.path.join(outfolder, f'y_{suffix}_{limit}.npy')}")

    return x, vocab_file


def preprocess_nebula_speakeasy(
        folder, 
        limit=None,
        outfolder=None,
        dump_paths=True,
        # filter settings
        record_fields=[
            'file_access.event',
            'file_access.path',
            'network_events.traffic.server',
            'network_events.traffic.port',
            'registry_access.event',
            'registry_access.path',
            'apis.api_name',
            'apis.args',
            'apis.ret_val',
        ],
        record_limits = {
            "network_events.traffic": 256
        },
        # tokenizer settings
        tokenizer_model=None,
        json_cleanup_symbols = ['"', "'", ":", ",", "[", "]", "{", "}", "\\", "/"],
        stopwords = ['api_name', 'args', 'ret_val', 'event', 'path', 'open_flags', 'access_flags', 'size', 'server', 'proto', 'port', 'method'],
        vocab_size = 50000,
        seq_len = 512,
        tokenizer_type="bpe"
):
    suffix = "test" if tokenizer_model else "train"
    truelimit = limit if limit else None
    limit = limit if limit else "full"
    assert tokenizer_type in ["bpe", "whitespace"], "tokenizer_type must be either 'bpe' or 'whitespace'"
    if outfolder is None:
        outfolder = f"nebula_speakeasy_{int(time.time())}"
    if not os.path.exists(outfolder):
        os.makedirs(outfolder, exist_ok=True)

    if os.path.exists(os.path.join(outfolder, f"x_{suffix}_{limit}.npy")):
        logging.warning(f" [!] Skipping since exists: {os.path.join(outfolder, f'x_{suffix}_{limit}.npy')}")
        with open(os.path.join(outfolder, f'y_names_{suffix}_{limit}.json')) as f:
            y_filepaths = orjson.loads(f.read())
        y = np.load(os.path.join(outfolder, f"y_{suffix}_{limit}.npy"))
        return None, y, y_filepaths

    # filter and normalize reports
    extractor = PEDynamicFeatureExtractor(
        speakeasyRecordFields=record_fields,
        recordLimits=record_limits
    )
    subfolders = [os.path.join(folder, x) for x in os.listdir(folder) if x.startswith("report_")]
    events, y, y_filepaths = read_and_filter_json_folders(
        subFolders=subfolders,
        filter_function=extractor.filter_and_normalize_report,
        benign_folders=["report_clean", "report_windows_syswow64"],
        limit=truelimit
    )
    y = np.array(y, dtype=np.int8)
    np.save(os.path.join(outfolder, f"y_{suffix}_{limit}.npy"), y)
    logging.warning(f" [!] Saved Y as {os.path.join(outfolder, f'y_{suffix}_{limit}.npy')}")
    
    if dump_paths:
        with open(os.path.join(outfolder, f"y_names_{suffix}_{limit}.json"), "w") as f:
            json.dump(y_filepaths, f, indent=4)
        logging.warning(f" [!] Saved Y names as {os.path.join(outfolder, f'y_names_{suffix}_{limit}.json')}")

    # train tokenizer
    if not tokenizer_model:
        if tokenizer_type == "bpe":
            tokenizer = JSONTokenizerBPE(
                vocab_size=vocab_size,
                seq_len=seq_len,
                cleanup_symbols=json_cleanup_symbols,
                stopwords=stopwords
            )
        else:
            tokenizer = JSONTokenizerWhiteSpace(
                vocab_size=vocab_size,
                seq_len=seq_len,
                cleanup_symbols=json_cleanup_symbols,
                stopwords=stopwords
            )

        logging.warning(" [*] Initializing tokenizer training...")
        tokenizer.train(
            events,
            vocab_size=vocab_size,
            model_prefix=os.path.join(outfolder, f"tokenizer_{vocab_size}")
        )
    else:
        if tokenizer_type == "bpe":
            tokenizer = JSONTokenizerBPE(
                vocab_size=vocab_size,
                seq_len=seq_len,
                model_path=tokenizer_model,
                cleanup_symbols=json_cleanup_symbols,
                stopwords=stopwords
            )
        else:
            tokenizer = JSONTokenizerWhiteSpace(
                vocab_size=vocab_size,
                seq_len=seq_len,
                cleanup_symbols=json_cleanup_symbols,
                stopwords=stopwords
            )
            tokenizer.load_vocab(tokenizer_model)
    
    logging.warning(" [*] Encoding and padding...")
    encoded = tokenizer.encode(events)
    x = tokenizer.pad_sequences(encoded, seq_len=seq_len)
    np.save(os.path.join(outfolder, f"x_{suffix}_{limit}.npy"), x)
    logging.warning(f" [!] Saved X as {os.path.join(outfolder, f'x_{suffix}_{limit}.npy')}")

    return x, y, y_filepaths


def preprocess_quovadis_speakeasy(
    report_paths,
    y=None,
    vocab_file=None,
    outfolder=None,
    seq_len=150,
    top_api=600,
    limit=None,
):
    suffix = "test" if vocab_file else "train"
    limit = limit if limit else "full"
    if outfolder is None:
        outfolder = f"neurlux_{int(time.time())}"
    if not os.path.exists(outfolder):
        os.makedirs(outfolder, exist_ok=True)

    if os.path.exists(os.path.join(outfolder, f"x_{suffix}_{limit}.npy")):
        logging.warning(f" [!] Skipping since exists: {os.path.join(outfolder, f'x_{suffix}_{limit}.npy')}")
        vocab_file = os.path.join(outfolder, [x for x in os.listdir(outfolder) if x.startswith("vocab")][0])
        return None, vocab_file

    logging.warning(" [*] Reading raw records...")
    reports_raw = []
    for file in tqdm(report_paths):
        with open(file) as f:
            reports_raw.append(orjson.loads(f.read()))

    if vocab_file:
        quovadis_model = QuoVadisModel(
            vocab=vocab_file,
            seq_len=seq_len
        )
        api_sequences = []
        for report in tqdm(reports_raw):
            api_sequences.append(report_to_apiseq(report))
    else:
        quovadis_model = QuoVadisModel(
            seq_len=seq_len
        )
        logging.warning(" [*] Building Quo Vadis vocabulary...")
        api_sequences = quovadis_model.build_vocab(reports_raw, top_api=top_api)
        vocab_file = quovadis_model.dump_vocab(outfolder)
    
    logging.warning(" [*] Encoding and padding reports...")
    x = quovadis_model.apisequences_to_arr(api_sequences)
    np.save(os.path.join(outfolder, f"x_{suffix}_{limit}.npy"), x)
    logging.warning(f" [!] Saved X as {os.path.join(outfolder, f'x_{suffix}_{limit}.npy')}")
    if y is not None:
        assert len(reports_raw) == y.shape[0], "X and Y must have the same length"
        np.save(os.path.join(outfolder, f"y_{suffix}_{limit}.npy"), y)
        logging.warning(f" [!] Saved Y as {os.path.join(outfolder, f'y_{suffix}_{limit}.npy')}")

    return x, vocab_file


def preprocess_nebula_cruparamer(
        events, 
        y,
        limit=None,
        outfolder=None,
        seq_len=512,
        vocab_size=50000,
        tokenizer_model=None,
        json_cleanup_symbols = None, #['"', "'", ":", ",", "[", "]", "{", "}", "\\", "/"],
        stopwords = [], #['api_name', 'args', 'ret_val', 'event', 'path', 'open_flags', 'access_flags', 'size', 'server', 'proto', 'port', 'method'],
):
    suffix = "test" if tokenizer_model else "train"
    limit = limit if limit else "full"

    if outfolder is None:
        outfolder = f"nebula_cruparamer_{int(time.time())}"
    if not os.path.exists(outfolder):
        os.makedirs(outfolder, exist_ok=True)

    if os.path.exists(os.path.join(outfolder, f"x_{suffix}_{limit}.npy")):
        logging.warning(f" [!] Skipping since exists: {os.path.join(outfolder, f'x_{suffix}_{limit}.npy')}")
        y = np.load(os.path.join(outfolder, f"y_{suffix}_{limit}.npy"))
        return None, y

    if not tokenizer_model:
        tokenizer = JSONTokenizerBPE(
            cleanup_symbols=json_cleanup_symbols,
            stopwords=stopwords
        )
        logging.warning(" [*] Initializing tokenizer training...")
        tokenizer.train(
            events,
            vocab_size=vocab_size,
            model_prefix=os.path.join(outfolder, f"tokenizer_{vocab_size}")
        )
    else:
        tokenizer = JSONTokenizerBPE(
            model_path=tokenizer_model,
            cleanup_symbols=json_cleanup_symbols,
            stopwords=stopwords
        )

    logging.warning(" [*] Encoding and padding...")
    encoded = tokenizer.encode(events)
    x = tokenizer.pad_sequences(encoded, sequenceLength=seq_len)
    np.save(os.path.join(outfolder, f"x_{suffix}_{limit}.npy"), x)
    logging.warning(f" [!] Saved X as {os.path.join(outfolder, f'x_{suffix}_{limit}.npy')}")
    
    y = np.array(y, dtype=np.int8)
    np.save(os.path.join(outfolder, f"y_{suffix}_{limit}.npy"), y)
    logging.warning(f" [!] Saved Y as {os.path.join(outfolder, f'y_{suffix}_{limit}.npy')}")
    
    return x, y


def preprocess_quovadis_cruparamer(
    api_sequences,
    y=None,
    vocab_file=None,
    outfolder=None,
    seq_len=150,
    top_api=600,
    limit=None,
):
    suffix = "test" if vocab_file else "train"
    limit = limit if limit else "full"
    if outfolder is None:
        outfolder = f"neurlux_{int(time.time())}"
    if not os.path.exists(outfolder):
        os.makedirs(outfolder, exist_ok=True)

    if os.path.exists(os.path.join(outfolder, f"x_{suffix}_{limit}.npy")):
        logging.warning(f" [!] Skipping since exists: {os.path.join(outfolder, f'x_{suffix}_{limit}.npy')}")
        vocab_file = os.path.join(outfolder, [x for x in os.listdir(outfolder) if x.startswith("vocab")][0])
        return None, vocab_file

    if vocab_file:
        quovadis_model = QuoVadisModel(
            vocab=vocab_file,
            seq_len=seq_len
        )
    else:
        quovadis_model = QuoVadisModel(
            seq_len=seq_len
        )
        logging.warning(" [*] Building Quo Vadis vocabulary...")
        api_sequences = quovadis_model.build_vocab(api_sequences, top_api=top_api, sequences=True)
        vocab_file = quovadis_model.dump_vocab(outfolder)

    logging.warning(" [*] Encoding and padding reports...")
    x = quovadis_model.apisequences_to_arr(api_sequences)
    np.save(os.path.join(outfolder, f"x_{suffix}_{limit}.npy"), x)
    logging.warning(f" [!] Saved X as {os.path.join(outfolder, f'x_{suffix}_{limit}.npy')}")
    if y is not None:
        assert len(api_sequences) == y.shape[0], "X and Y must have the same length"
        np.save(os.path.join(outfolder, f"y_{suffix}_{limit}.npy"), y)
        logging.warning(f" [!] Saved Y as {os.path.join(outfolder, f'y_{suffix}_{limit}.npy')}")

    return x, vocab_file
