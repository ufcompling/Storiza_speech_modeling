import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Union

from scripts.audio_analysis.MFA_Audio_Preprocesser import generate_mfa_items
import re
from collections import defaultdict

_NUM_RE = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$')

class MFAIpaAligner:
    """
    Thin wrapper around Montreal Forced Aligner for IPA workflows.

    - Default model/dict: english_us_mfa (IPA phone set).
    - For items with a provided IPA (string or list of strings), we create a tiny per-item dictionary that maps
      the transcript token 'CAND' to those IPA variants; MFA will pick the best-fitting variant.
    - For items without IPA, we rely on the pretrained english_us_mfa dictionary.
    - We parse the JSON output to get phone timings and reconstruct the winning IPA sequence.
    """

    def __init__(
            self,
            acoustic_model: str = "english_mfa",
            dictionary: str = "english_mfa",  # <-- now expected to be a FILE PATH if you want auto-merge
            mfa_bin: str = "mfa",
            temp_root: Union[str, Path] = "./temp",
            speaker_characters: Optional[Union[int, str]] = None,
            keep_original_filenames: bool = False,
    ):
        self.acoustic_model = acoustic_model
        self.dictionary = dictionary
        self.mfa_bin = mfa_bin
        self.temp_root = Path(temp_root)
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.speaker_characters = speaker_characters
        self.keep_original_filenames = keep_original_filenames or (speaker_characters is not None)

    def _resolve_base_dictionary_path(self) -> Optional[Path]:
        p = Path(self.dictionary).expanduser()
        if p.exists() and p.is_file():
            return p
        return None  # Caller will handle the 'model-name' case explicitly

    def _read_base_pron_map(self, dict_path: Union[str, Path]) -> Dict[str, List[str]]:
        """
        Load a (possibly weighted) MFA dictionary and return:
          {word_lower: ["ph1 ph2 ...", "ph1 ph2 ...", ...]}
        Any numeric weight columns are stripped so all variants are unweighted.
        """
        p = Path(dict_path).expanduser()
        prons = defaultdict(list)
        if not p.exists():
            return prons
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(";"):
                    continue
                parts = line.split()
                word = parts[0].lower()
                tail = parts[1:]
                # drop any leading numeric columns (pron probs, durations, etc.)
                i = 0
                while i < len(tail) and _NUM_RE.match(tail[i]):
                    i += 1
                phones = tail[i:]
                if phones:
                    prons[word].append(" ".join(phones))
        return prons

    def _load_alignment_analysis(self, out_dir: Path) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Reads alignment_analysis.csv if present and returns:
          { <stem>: {
              "speaker": str|None,
              "overall_log_likelihood": float|None,
              "speech_log_likelihood": float|None,
              "phone_duration_deviation": float|None,
              "snr": float|None
            }, ... }
        Keys are normalized to the audio/TextGrid JSON base *stem*.
        """
        csv_path = Path(out_dir) / "alignment_analysis.csv"
        if not csv_path.exists():
            return {}

        def norm_key(s: str) -> str:
            return s.strip().lower().replace(" ", "_")

        scores: Dict[str, Dict[str, Optional[float]]] = {}
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Try common column names MFA uses across versions
                file_field = row.get("file") or row.get("path") or row.get("Filename") or row.get("Sound file")
                stem = Path(file_field).stem if file_field else None
                if not stem:
                    continue

                # Normalize headers so we can tolerate format drift
                row_norm = {norm_key(k): v for k, v in row.items()}

                # Speaker may be present; otherwise we can derive later if needed
                speaker = row_norm.get("speaker")

                # Likelihoods & deviation (handle both snake_case and title case variants)
                overall_ll = (
                        row_norm.get("overall_log_likelihood")
                        or row_norm.get("alignment_log_likelihood")
                        or row_norm.get("log_likelihood")
                )
                speech_ll = (
                        row_norm.get("speech_log_likelihood")
                        or row_norm.get("speech_loglike")
                )
                phone_dev = (
                        row_norm.get("phone_duration_deviation")
                        or row_norm.get("duration_deviation")
                )

                # SNR can appear in a few forms
                snr_val = (
                        row_norm.get("snr")
                        or row_norm.get("snr_db")
                        or row.get("SNR (dB)")
                        or row.get("SNR")
                )

                def to_float(x):
                    try:
                        return float(x)
                    except (TypeError, ValueError):
                        return None

                scores[stem] = {
                    "speaker": speaker if speaker not in ("", None) else None,
                    "overall_log_likelihood": to_float(overall_ll),
                    "speech_log_likelihood": to_float(speech_ll),
                    "phone_duration_deviation": to_float(phone_dev),
                    "snr": to_float(snr_val),
                }
        return scores

    @staticmethod
    def _hash_items(items: List[Dict]) -> str:
        # Hash the minimal info that should change the output
        h = hashlib.sha1()
        for it in items:
            word = (it.get("word") or "").strip()
            ipa = it.get("ipa")
            if isinstance(ipa, list):
                ipa_str = " || ".join(map(str, ipa))
            else:
                ipa_str = str(ipa) if ipa is not None else ""
            audio = (it.get("audio") or "").strip()
            h.update(word.encode("utf-8"))
            h.update(b"\0")
            h.update(ipa_str.encode("utf-8"))
            h.update(b"\0")
            h.update(audio.encode("utf-8"))
            h.update(b"\0")
        return h.hexdigest()[:16]

    def _run(self, args: List[str], cwd: Optional[Union[str, Path]] = None):
        """Run a subprocess, raise on error with nice message."""
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Command failed ({proc.returncode}):\n{' '.join(args)}\n--- OUTPUT ---\n{proc.stdout}"
            )
        return proc.stdout

    def _maybe_speaker_chars_flag(self) -> List[str]:
        """Return ['--speaker_characters', value] if configured, else []."""
        if self.speaker_characters is None:
            return []
        # MFA accepts an integer (e.g., 8) or the special string 'prosodylab'
        # If user gave a string (like 'prosodylab'), pass it directly; otherwise cast int to str.
        val = str(self.speaker_characters)
        return ["--speaker_characters", val]

    def _prepare_item_corpus(
        self,
        work_dir: Path,
        item_id: str,
        word: str,
        audio_path: Union[str, Path],
        ipa: Optional[Union[str, List[str]]],
    ):
        corpus_dir = work_dir / f"{item_id}_corpus"
        out_dir = work_dir / f"{item_id}_aligned"
        dict_path = None

        corpus_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Choose target filename: either keep original basename or use item_id
        if self.keep_original_filenames:
            target_name = audio_path.name  # preserve prefix for --speaker_characters
        else:
            target_name = f"{item_id}{audio_path.suffix}"

        target_audio = corpus_dir / target_name
        shutil.copy2(audio_path, target_audio)

        # Transcript token
        transcript_token = f"CAND_{word}" if ipa is not None else word

        # .lab must match the audio basename (sans extension)
        (corpus_dir / f"{Path(target_name).stem}.lab").write_text(
            transcript_token + "\n", encoding="utf-8"
        )

        # Optional per-item dictionary
        if ipa is not None:
            dict_path = work_dir / f"{item_id}_dict.txt"
            with dict_path.open("w", encoding="utf-8") as f:
                variants = ipa if isinstance(ipa, list) else [ipa]
                for v in variants:
                    phones = " ".join(v.split())
                    f.write(f"CAND_{word}  {phones}\n")

        return corpus_dir, out_dir, dict_path

    @staticmethod
    def _parse_json_alignment(json_path: Path):
        """
        Parse MFA's JSON export (structure can vary slightly by version).
        Returns both phones and words.
        """
        data = json.loads(json_path.read_text(encoding="utf-8"))

        phones = []
        words = []

        # --- Phones ---
        if "tiers" in data:
            tiers = data["tiers"]
            if isinstance(tiers, dict):
                # Phones
                if "phones" in tiers and isinstance(tiers["phones"], dict):
                    entries = tiers["phones"].get("entries", [])
                    phones = [
                        {"phone": e[2], "start": float(e[0]), "end": float(e[1])}
                        for e in entries if len(e) == 3
                    ]
                # Words
                if "words" in tiers and isinstance(tiers["words"], dict):
                    entries = tiers["words"].get("entries", [])
                    words = [
                        {"word": e[2], "start": float(e[0]), "end": float(e[1])}
                        for e in entries if len(e) == 3
                    ]

        # Filter out silences commonly used by MFA
        SILS = {"sil", "sp", "spn", "silence", ""}
        phones = [p for p in phones if p["phone"] not in SILS]
        words = [w for w in words if w["word"] not in SILS]

        return {"phones": phones, "words": words}

    def align(
        self,
        items: List[Dict[str, Union[str, List[str]]]],
        output_dir: Union[str, Path],
        batch=True,
        debug=True
    ) -> List[Dict]:
        cache_key = self._hash_items(items)
        session_dir = self.temp_root / cache_key
        session_dir.mkdir(parents=True, exist_ok=True)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if batch:
            corpus_dir = session_dir / "batch_corpus"
            out_dir = session_dir / "batch_aligned"
            corpus_dir.mkdir(parents=True, exist_ok=True)
            out_dir.mkdir(parents=True, exist_ok=True)

            token_map = {}
            combined_dict = corpus_dir / "combined_dict.txt"

            # NEW: load base pronunciations once, but stripped (unweighted)
            base_dict_path = self._resolve_base_dictionary_path()
            base_prons = self._read_base_pron_map(base_dict_path) if base_dict_path else {}

            missing_from_base = set()

            with combined_dict.open("w", encoding="utf-8") as dict_f:
                for idx, it in enumerate(items):
                    word = (it["word"] or "").strip()
                    low = word.lower()
                    ipa = it.get("ipa")
                    audio_path = Path(it["audio"])
                    target_name = audio_path.name if self.keep_original_filenames else f"utt{idx:04d}{audio_path.suffix}"
                    shutil.copy2(audio_path, corpus_dir / target_name)

                    # Always use a safe per-item token
                    token_key = f"C{idx:04d}"
                    (corpus_dir / f"{Path(target_name).stem}.lab").write_text(token_key + "\n", encoding="utf-8")

                    if ipa:
                        # Respect user-provided IPA(s)
                        variants = ipa if isinstance(ipa, list) else [ipa]
                        for v in variants:
                            dict_f.write(f"{token_key}  {' '.join(str(v).split())}\n")
                    else:
                        # Pull *all* base variants for this word, unweighted
                        var_list = base_prons.get(low, [])
                        if var_list:
                            for v in var_list:
                                dict_f.write(f"{token_key}  {v}\n")
                        else:
                            # Fallback: let MFA use base dict via the orthographic token (rare)
                            # Comment these three lines if you prefer to *not* fall back.
                            fallback_token = re.sub(r"[^A-Za-z']+", "", low) or f"W{idx:04d}"
                            (corpus_dir / f"{Path(target_name).stem}.lab").write_text(fallback_token + "\n",
                                                                                      encoding="utf-8")
                            missing_from_base.add(fallback_token)

                    token_map[Path(target_name).stem] = word

            # IMPORTANT: if we used any fallback orthographic tokens, we must append the base dict
            if missing_from_base and base_dict_path:
                with open(base_dict_path, "r", encoding="utf-8") as base_f, \
                        open(combined_dict, "a", encoding="utf-8") as dict_f:
                    dict_f.write("\n")
                    dict_f.write(base_f.read())

            if debug:
                print(f"[INFO] Running MFA on {len(items)} items...")

            cmd = [
                self.mfa_bin, "align",
                str(corpus_dir),
                str(combined_dict),
                self.acoustic_model,
                str(out_dir),
                "--output_format", "json",
                "--clean",
                "--always_verbose",
                "--fine_tune",
                "--config_path", "./decoder_config.yaml"

            ]
            cmd += self._maybe_speaker_chars_flag()
            self._run(cmd)

            analysis_scores = self._load_alignment_analysis(out_dir)

            # Parse JSONs
            json_files = list(out_dir.rglob("*.json"))
            results = []
            for jpath in json_files:
                parsed = self._parse_json_alignment(jpath)
                phones = parsed["phones"]
                best_ipa = " ".join(p["phone"] for p in phones)
                base = Path(jpath).stem
                orig_word = token_map.get(base, "")
                sc = analysis_scores.get(base, {})

                # If CSV lacked 'speaker', derive it from filename when possible
                speaker_val = sc.get("speaker")
                if speaker_val is None and self.keep_original_filenames and isinstance(self.speaker_characters, int):
                    # The audio filename in batch mode is either original or "utt####"
                    # If original, derive speaker as the first N chars of the basename (minus extension).
                    basename = Path(jpath.with_suffix(".wav")).name  # JSON stem with .wav
                    stem_for_speaker = Path(basename).stem
                    speaker_val = stem_for_speaker[: self.speaker_characters]

                results.append({
                    "word": orig_word,
                    "best_ipa": best_ipa,
                    "audio": str(jpath.with_suffix(".wav")),
                    "phone_segments": phones,
                    "output_json": str(jpath),
                    "work_dir": str(corpus_dir),
                    # --- NEW FIELDS ---
                    "speaker": speaker_val,
                    "overall_log_likelihood": sc.get("overall_log_likelihood"),
                    "speech_log_likelihood": sc.get("speech_log_likelihood"),
                    "phone_duration_deviation": sc.get("phone_duration_deviation"),
                    "snr": sc.get("snr"),
                })

            if debug:
                print(f"[INFO] Alignment complete. Parsed {len(results)} results.")
            return results

        # -------- non-batch path (tiny fixes only) --------
        results = []
        for idx, it in enumerate(items):
            word = it.get("word")
            ipa = it.get("ipa")
            audio = it.get("audio")
            if not word or not audio:
                raise ValueError("Each item must include 'word' and 'audio'. 'ipa' is optional.")

            item_id = f"utt{idx:04d}"
            work_dir = session_dir / item_id
            work_dir.mkdir(parents=True, exist_ok=True)

            corpus_dir, out_dir, dict_path = self._prepare_item_corpus(
                work_dir=work_dir, item_id=item_id, word=word, audio_path=audio, ipa=ipa
            )

            # validate
            val_cmd = [self.mfa_bin, "validate", str(corpus_dir)]
            if dict_path is not None:
                val_cmd += [str(dict_path), self.acoustic_model]
            else:
                # If no per-item dict, require that `self.dictionary` is usable by MFA directly
                val_cmd += [self.dictionary, self.acoustic_model]
            val_cmd += self._maybe_speaker_chars_flag()
            self._run(val_cmd)

            # align
            if dict_path is not None:
                cmd = [
                    self.mfa_bin, "align",
                    str(corpus_dir),
                    str(dict_path),
                    self.acoustic_model,
                    str(out_dir),
                    "--output_format", "json",
                    "--num_jobs", "16",   # FIX: missing comma before
                    "--clean",
                ]
            else:
                cmd = [
                    self.mfa_bin, "align",
                    str(corpus_dir),
                    self.dictionary,
                    self.acoustic_model,
                    str(out_dir),
                    "--output_format", "json",
                    "--clean",
                ]
            cmd += self._maybe_speaker_chars_flag()
            self._run(cmd)

            json_files = list(out_dir.rglob("*.json"))
            if not json_files:
                raise FileNotFoundError(f"No JSON found in {out_dir}. Ensure --output_format json is supported by your MFA version.")
            json_path = json_files[0]
            parsed = self._parse_json_alignment(json_path)
            phones = parsed["phones"]
            best_ipa = " ".join(p["phone"] for p in phones)

            item_out_dir = output_dir / item_id
            if item_out_dir.exists():
                shutil.rmtree(item_out_dir)
            shutil.copytree(out_dir, item_out_dir)

            results.append({
                "word": word,
                "best_ipa": best_ipa,
                "audio": str(audio),
                "phone_segments": phones,
                "work_dir": str(work_dir),
                "output_json": str(json_path),
            })
        return results




if __name__=="__main__":
    aligner = MFAIpaAligner(
        acoustic_model="english_mfa",
        dictionary="/home/michaelbennie/Documents/MFA/pretrained_models/dictionary/english_us_mfa.dict",
        mfa_bin="mfa",
        temp_root="./temp",
        speaker_characters=36,  # <- set to the exact length of the UID prefix at the start of the filename
        keep_original_filenames=True,  # <- ensure MFA can see that prefix
    )

    CSV_PATH = Path("../../processed_annotations/word_level_data.csv")
    audio_root = Path(
        "~/OneDrive/Leite,Walter's files - Storiza Corpus Spring 2025/Processed_Utterances/annotated_word_segments"
    ).expanduser()

    items = generate_mfa_items(
        csv_path=CSV_PATH,
        directory_root=audio_root,  # NOT one level above
        item_range=[235, 235+1],
        strip_filename=True  # since the CSV paths already include the filenames
    )



    results = aligner.align(items, output_dir="./aligned_outputs")

    for r in results:
        print("WORD:", r["word"])
        print("BEST IPA:", r["best_ipa"])
        print("AUDIO:", r["audio"])
        print("PHONES:")
        for ph in r["phone_segments"]:
            print(f"  {ph['start']:.3f}–{ph['end']:.3f}  {ph['phone']}")
        print("Artifacts:", r["work_dir"])
        for val in ["speaker","overall_log_likelihood","speech_log_likelihood","phone_duration_deviation","snr"]:
            print(val.upper()+":", r[val])
        print()


    items.append({'audio': "/home/michaelbennie/OneDrive/Leite,Walter's files - Storiza Corpus Spring 2025/Processed_Utterances/annotated_word_segments/uid_gUfM3yODstSdwltidtEzSlIo0q93_sid_YJYpGQOGORLj0onYXX3W_1742770181_195798254_word_segment_13.wav", 'ipa': "ej", 'word': 'a_1'})
    results = aligner.align(items, output_dir="./aligned_outputs")

    for r in results:
        print("WORD:", r["word"])
        print("BEST IPA:", r["best_ipa"])
        print("AUDIO:", r["audio"])
        print("PHONES:")
        for ph in r["phone_segments"]:
            print(f"  {ph['start']:.3f}–{ph['end']:.3f}  {ph['phone']}")
        print("Artifacts:", r["work_dir"])
        for val in ["speaker","overall_log_likelihood","speech_log_likelihood","phone_duration_deviation","snr"]:
            print(val.upper()+":", r[val])
        print()

    items.append({'audio': "/home/michaelbennie/OneDrive/Leite,Walter's files - Storiza Corpus Spring 2025/Processed_Utterances/annotated_word_segments/uid_gUfM3yODstSdwltidtEzSlIo0q93_sid_YJYpGQOGORLj0onYXX3W_1742770181_195798254_word_segment_13.wav", 'ipa': "ə", 'word': 'a_2'})
    results = aligner.align(items, output_dir="./aligned_outputs")

    for r in results:
        print("WORD:", r["word"])
        print("BEST IPA:", r["best_ipa"])
        print("AUDIO:", r["audio"])
        print("PHONES:")
        for ph in r["phone_segments"]:
            print(f"  {ph['start']:.3f}–{ph['end']:.3f}  {ph['phone']}")
        print("Artifacts:", r["work_dir"])
        for val in ["speaker","overall_log_likelihood","speech_log_likelihood","phone_duration_deviation","snr"]:
            print(val.upper()+":", r[val])
        print()