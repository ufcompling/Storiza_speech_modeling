import os
from pathlib import Path
from typing import Union, List, Optional, Any, Dict
import riva.client
import grpc

import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Read API key from environment
api_key = os.getenv("NVIDIA_API_KEY")
input_file="./../raw_data/audio/uid_0FdMSMtn95PJ9tLFeW3F4sFLPMh1_sid_GcbJYqscm9YpPDCHkJWP_1743044185.wav"


def transcribe_file_offline_full(
    server: str,
    api_key: str,
    input_file: str | Path,
    *,
    ssl_cert: Optional[str] = None,
    use_ssl: bool = True,
    list_models: bool = False,
    language_code: str = "en-US",
    max_alternatives: int = 1,
    profanity_filter: bool = False,
    automatic_punctuation: bool = True,
    no_verbatim_transcripts: bool = False,
    word_time_offsets: bool = False,
    speaker_diarization: bool = False,
    diarization_max_speakers: Optional[int] = None,
    boosted_lm_words: Optional[List[str]] = None,
    boosted_lm_score: Optional[float] = None,
        start_history: Optional[int] = 0,
    start_threshold: Optional[float] = 0.0,
    stop_history: Optional[int] = 0,
    stop_history_eou: Optional[int] = 0,
    stop_threshold: Optional[float] = 0.0,
    stop_threshold_eou: Optional[float] = 0.0,

    custom_configuration: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Full-featured Riva offline ASR transcriber and model lister.

    If list_models is True, prints the models and returns None.
    Otherwise, returns transcript string or raises exception.
    """
    input_path = Path(input_file).expanduser().resolve() if input_file else None
    # Auth and ASRService setup

    function_id: str = "d3fe9151-442b-4204-a70d-5fcc597fd610"

    auth = riva.client.Auth(ssl_cert, use_ssl, server, [
            ("function-id", function_id),
            ("authorization", f"Bearer {api_key}")
        ],)
    asr_service = riva.client.ASRService(auth)


    if list_models:
        asr_models = dict()
        config_response = asr_service.stub.GetRivaSpeechRecognitionConfig(
            riva.client.proto.riva_asr_pb2.RivaSpeechRecognitionConfigRequest()
        )
        for model_config in config_response.model_config:
            if model_config.parameters["type"] == "offline":
                language_code = model_config.parameters['language_code']
                model = {"model": [model_config.model_name]}
                if language_code in asr_models:
                    asr_models[language_code].append(model)
                else:
                    asr_models[language_code] = [model]
        print("Available ASR models")
        asr_models = dict(sorted(asr_models.items()))
        print(asr_models)
        return None
    # Validate audio file
    if not input_path or not os.path.isfile(input_path):
        raise FileNotFoundError(f"Invalid input file path: {input_path}")
    # Build config
    config = riva.client.RecognitionConfig(
        language_code=language_code,
        max_alternatives=max_alternatives,
        profanity_filter=profanity_filter,
        enable_automatic_punctuation=automatic_punctuation,
        verbatim_transcripts=not no_verbatim_transcripts,
        enable_word_time_offsets=word_time_offsets or speaker_diarization,
    )
    riva.client.add_word_boosting_to_config(config, boosted_lm_words, boosted_lm_score)
    riva.client.add_speaker_diarization_to_config(config, speaker_diarization, diarization_max_speakers)
    riva.client.add_endpoint_parameters_to_config(
        config,
        start_history,
        start_threshold,
        stop_history,
        stop_history_eou,
        stop_threshold,
        stop_threshold_eou
    )
    with open(input_path, 'rb') as fh:
        data = fh.read()
    try:
        response = asr_service.offline_recognize(data, config)
    except grpc.RpcError as e:
        raise RuntimeError(f"Riva ASR error: {e.details()}") from e
    # Collect transcript
    def _collect_transcripts(results) -> List[str]:
        return [r.alternatives[0].transcript for r in results if r.alternatives]
    transcript_parts = _collect_transcripts(response.results)
    return " ".join(transcript_parts).strip()

# Example usage (uncomment to test)
if __name__ == "__main__":
    SERVER = "grpc.nvcf.nvidia.com:443"
    API_KEY = "<your‑bearer‑token>"
    print(transcribe_file_offline_full(SERVER, api_key, input_file))
