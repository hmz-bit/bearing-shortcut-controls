DATA SETS, LICENCES AND HOW TO OBTAIN THEM
===========================================

No raw data, no derived signal arrays and no third-party split or feature files are included in this repository.
Everything below has to be fetched from the original provider.  All code expects the fetched material under the
repository root (the folder that contains scripts/ and results/):

    data/                          raw and cached material for the five data sets (not versioned)
    bearing-data-leakage/          local clone of the third-party benchmark repository (splits, features, saved logits)

Statements about licences were taken from the notes kept during development; where the note records only
the licence name, only the name is given here.  Check the provider's page before redistributing anything.


1. Third-party benchmark repository (splits, features, saved logits)
-----------------------------------------------------------------
Vieira et al., "Towards a more realistic evaluation of machine learning models for bearing fault diagnosis",
Mechanical Systems and Signal Processing 258 (2026) 114640, DOI 10.1016/j.ymssp.2026.114640 (citation metadata as
given in the CITATION.cff of that repository).

    URL:     https://github.com/gama-ufsc/bearing-data-leakage
    Commit:  4c9fd8cb3cf9d5aa4ef2b653cb314a0727e12f3d   (the HEAD of the local clone used for all experiments;
             commit date Wed Jul 8 14:18:32 2026 -0300; read with "git log -1")
    Licence: none (the clone contains no LICENSE file).  Therefore NOTHING from this repository is copied here:
             no code, no split files, no feature files, no saved logits, no paper PDFs.

What this project reads from it, as data only (no code of theirs is imported or executed, nothing is unpickled):
    bearing-data-leakage/data/features/paderborn_features_segmented.parquet      (15 hand-crafted PU features)
    bearing-data-leakage/data/splits/paderborn/{tuning,eval}/run_*.parquet      (PU bearing-wise splits)
    bearing-data-leakage/data/splits/cwru/eval/*.parquet                          (CWRU splits)
    bearing-data-leakage/data/splits/uored-vafcls/eval/run_*.parquet              (UORED splits)
    bearing-data-leakage/data/saved_logits/pu/*/test/epoch_30_run_0.pt            (their published B1 logits on PU)
Their configuration and model files were read as text to re-implement the B1 baseline (scripts/b1); the files
consulted are listed in the module docstrings of scripts/b1/core.py, envspec.py and wdcnn.py.

To obtain it:      tools/fetch_third_party.sh        (git clone, then git checkout of the commit above)

Derived numbers that depend on their saved logits (the PU "B1" column of the cross-data-set table) are reproduced by
scripts/evaltools/reproduce_b1.py and pu_data.logits_to_long after the clone is in place.  The window-level score file
scripts/evaltools/results/b1_predictions_long.parquet is derived from their logits and is therefore not part of this repository.
Per-split summary tables that compare own and published B1 values (results/b1/val_*published*.csv, val_3seed_table.csv) are
included as numerical results only.


2. Paderborn bearing data (PU)
------------------------------
Provider:  Chair of Design and Drive Technology, Paderborn University, "KAt" Bearing Data Center.
URL:       https://groups.uni-paderborn.de/kat/BearingDataCenter/     (archive URLs are used by scripts/download_pu.sh)
Licence:   CC BY-NC 4.0 (research use; commercial use requires the data owner's permission).
Citation:  Lessmeier et al. 2016 (the reference named in scripts/download_pu.sh and in the comments of scripts/lfpncn/geometry.py;
           Table 1 of that paper is the source of the bearing geometry used).  The full bibliographic entry was not
           re-checked while preparing this repository; use the entry given on the provider's page.
Used:      32 bearing archives (K001-K006, KA01 KA03-KA09 KA15 KA16 KA22 KA30, KB23 KB24 KB27, KI01 KI03-KI05 KI07 KI08
           KI14 KI16-KI18 KI21); the experiments use the 17 bearings of the benchmark (K001-K006, KA04 KA15 KA16 KA22 KA30,
           KI04 KI14 KI16 KI17 KI18 KI21) plus 12 artificially damaged bearings (KA01 KA03 KA05-KA09, KI01 KI03 KI05 KI07 KI08);
           the three bearings with combined damage (KB*) are excluded.
Obtain:    scripts/download_pu.sh (downloads and extracts into data/pu), then
           python -I scripts/build_pu_metadata.py ; python -I scripts/build_cache.py   (window cache data/cache/pu_windows.npz).
Operating conditions in the file names (N{rpm}_M{torque}_F{force}): N15_M07_F10 (1500 rpm, 0.7 Nm, 1000 N), N09_M07_F10 (900 rpm),
N15_M01_F10 (0.1 Nm), N15_M07_F04 (400 N).  Bearing geometry used: 6203, 8 elements, element diameter 6.75 mm, pitch circle diameter
28.55 mm, contact angle 0.


3. CWRU bearing data
--------------------
Provider:  Case Western Reserve University Bearing Data Center, official 12 kHz drive-end and fan-end fault data pages
           (engineering.case.edu bearing data center; the two table pages were saved as 12k-drive-end-bearing-fault-data.html and
           12k-fan-end-bearing-fault-data.html and their link targets used to map waveform ids to file numbers).
Licence:   no licence statement was recorded in the development notes and none was found in the saved pages; treat as publicly
           available data with unspecified terms and follow the provider's citation request.
Used:      72 official 12 kHz files (144 waveforms = drive-end and fan-end channel of each file), split ids from the
           third-party repository (section 1).
Obtain:    scripts/cwru/download_cwru.py downloads the 72 files, but it reads data/cwru/pages/cwru_links.json, a table of the link
           targets of the two saved pages.  The script that produced that json file is not part of the development record; rebuild
           it by extracting the href of every .mat link of the two pages into {"DE" or "FE": {label: url}} with labels such as
           IR007_0, B014_2, OR007@6_1 (NOT VERIFIED: format inferred from scripts/cwru/download_cwru.py).
           Then: python -I scripts/cwru/validate_raw.py ; scripts/cwru/speed_check_cwru.py ; scripts/cwru/build_cwru_cache.py.


4. UORED-VAFCLS (University of Ottawa constant load and speed rolling-element bearing vibration and acoustic fault signatures)
-----------------------------------------------------------------------------------------------------------------
Provider:  Mendeley Data, DOI 10.17632/y2px5tg92h, version 5 (the version used; "uored_v5.zip" in the local copy).
Licence:   CC BY 4.0.
Citation:  Sehri M., Dumond P., Bouchard M., "University of Ottawa constant load and speed rolling-element bearing vibration and
           acoustic fault signature datasets", Data in Brief 49 (2023) 109327 (open access, PMC10331275).
Used:      the accelerometer column of the csv files; 20 physical bearings, three recordings each (healthy, developing fault, faulty).
How the faults arose (checked against the full text of the data descriptor; short quotes):
  - "Bearings are run from healthy to failure."
  - "The drive end bearing seals were removed and the bearings were degreased to accelerate deterioration during testing."
    "They were then replaced after each test."  Twenty tests: the first five with NSK 6203ZZ, the next fifteen with FAFNIR 203KD,
    both with pitch diameter 28.50 mm, ball diameter 6.77 mm and 8 balls.
  - "Each set of bearing data is selected from a pool of 50 files collected over time"; healthy data "selected within the first 5 files",
    developing-fault data "selected within files 25 to 30" (the half-life point of the bearing), faulty data "selected within files 45 to 50".
  - "The faults were both identified visually and validated with the plotted FFTs."
  - Healthy, inner, outer and cage recordings were taken "under a consistent nominal load of 400 N"; for ball-fault data "no load was applied
    because ball faults developed naturally within a reasonable amount of time under degreased conditions".
  - Nothing in the descriptor describes machined, drilled or otherwise seeded defects.  The faults are therefore treated here as naturally
    developed damage produced by accelerated run-to-failure; the three states of a bearing are life stages of one physical bearing.
    (A secondary source that describes UORED as "artificial and naturally developed" could not be reconciled with the descriptor.)
  - The recorded speed column is one value per file and disagrees with the shaft line of the vibration spectrum for most files; see
    results/uored/speed_check.txt.  A single constant shaft frequency (29.75 Hz) was therefore used (protocols/PREDECLARED_UORED.txt item 3).
  - The descriptor gives a 400 N load; the third-party repository reports 500 N (not reconciled).
Obtain:    download the archive from the Mendeley Data page, extract the csv files into data/uored/csv/, then
           scripts/uored/speed_check.py, scripts/uored/build_uored_cache.py (cache), scripts/uored/build_b1_bank.py (B1 spectra).


5. XJTU-SY accelerated-life bearing data set
--------------------------------------------
Provider:  Xi'an Jiaotong University and Changxing Sumyoung Technology; data page and mirrors listed at biaowang.tech/xjtu-sy-bearing-datasets
           (mirrors: Google Drive, Dropbox, MediaFire, MEGA, Baidu as noted during development; automated download of the Dropbox link
           returns an HTML page, so the files have to be downloaded manually).
Licence:   none stated.  The data page says the data are publicly available for validating prognostics algorithms and asks for citation.
Citation:  Wang B., Lei Y., Li N., Li N., "A hybrid prognostics approach for estimating remaining useful life of rolling element bearings",
           IEEE Transactions on Reliability 69(1):401-412, 2020.
Facts used (taken from the data providers' own documents; see protocols/PREDECLARED_XJTU.txt "SOURCE FACTS"):
  15 bearings (type LDK UER204) in three conditions of five bearings (35 Hz / 12 kN, 37.5 Hz / 11 kN, 40 Hz / 10 kN); two PCB 352C33
  accelerometers (horizontal and vertical), 25.6 kHz, 1.28 s (32768 samples) every minute, one csv per sample (column 1 horizontal);
  failure when the amplitude exceeds 20 g.  Failure locations per bearing (table 3 of the providers' interpretation document):
  1_1 outer, 1_2 outer, 1_3 outer, 1_4 cage, 1_5 inner and outer, 2_1 inner, 2_2 outer, 2_3 cage, 2_4 outer, 2_5 outer, 3_1 outer,
  3_2 inner, rolling element, cage and outer, 3_3 inner, 3_4 inner, 3_5 outer.
  Not verified from a primary source: the geometry (D = 34.55 mm, d = 7.92 mm, z = 8, contact angle 0 deg) was taken from the providers' document;
  an earlier note recorded it only from a secondary source.
Used:      only the first and last 10% of the files of every bearing (at least 5 files per end; 1862 files).
Obtain:    download the archive volumes manually and extract the first and last files of every bearing into
           data/xjtu/raw_selected/XJTU-SY_Bearing_Datasets/<condition folder>/<bearing>/<n>.csv, and write
           data/xjtu/selection_first_last_10pct.json (read by scripts/xjtu/xjtu_common.py).  The code that extracted the files and
           wrote that json file is NOT part of the recorded scripts (it was run interactively), so its exact format is not documented here;
           the selection rule is in protocols/PREDECLARED_XJTU.txt item 2.  Then scripts/xjtu/speed_check_xjtu.py and
           scripts/xjtu/build_windows.py.


6. Paderborn run-to-failure data under time-varying speed and load
------------------------------------------------------------------
Provider:  Zenodo record 10868257 (concept DOI 10.5281/zenodo.10805042); 17 experiments B01-B17 (one ball bearing 61806-2RS each);
           descriptor PDF by Aimiyekagbon (v2, 2024-03-25).
Licence:   CC BY 4.0.
Used:      the first second of channel accHorizRear_A of selected snapshots (first and last 10% of the in-range snapshots per experiment,
           at most 100 per end).  Not verified from the descriptor: the contact angle (0 is assumed).
Obtain:    scripts/rtf/list_manifest.py (reads the zip central directories over HTTP range requests), scripts/rtf/fetch_aux.py,
           scripts/rtf/speed_check_rtf.py, scripts/rtf/select_snapshots.py, scripts/rtf/build_rtf_windows.py (reads only the selected members with
           HTTP range requests from the official Zenodo API URLs; the full archive set is about 152 GB and is not downloaded).


7. KAIST bearing data set, part 1 (constant speed), supplementary
-----------------------------------------------------------------
Provider:  Korea Advanced Institute of Science and Technology; Mendeley Data record ztmf3m7h5x (version 6); data article Jung et al., Data in Brief 48 (2023) 109049.
Licence:   CC BY 4.0 (licence object of the Mendeley record).
Facts used (data article): NSK 6205 DDU, 9 balls, ball diameter 7.90 mm, pitch diameter 38.5 mm, contact angle 0 deg; 3010 rpm (50.17 Hz); vibration at 25.6 kHz, 60 s per faulty
           recording; four accelerometer channels (x and y at housing A and B), the fault-seeded bearing in housing A; inner- and outer-race faults with crack sizes 0.3, 1.0 and 3.0 mm
           (damage is artificial); loads 0, 2 and 4 Nm.  Not stated in the article: whether each crack size is a separate bearing, and the order and dates of the recordings.
Used:      18 files {0,2,4}Nm_{BPFI,BPFO}_{03,10,30}.mat (about 722 MB compressed in the 2.7 GB archive vibration.zip), channel column 0 (x direction, housing A; the files do not store channel
           names, the mapping follows the article's channel order, protocols/PREDECLARED_KAIST.txt addendum 1).
Obtain:    scripts/kaist/fetch_kaist.py (reads the 18 members of vibration.zip with HTTP range requests from the public link, checks the CRC-32 of each member),
           scripts/kaist/run_kaist.py build, then scripts/kaist/run_kaist.py run.


8. Files derived from these data that ARE in this repository
------------------------------------------------------------
results/ contains model predictions (scores and labels per window, bearing identifiers), per-split and per-bearing metrics, bootstrap outputs and
logs produced by this project.  Bearing identifiers and window indices are those of the providers' file naming.  No signal samples are included.
Providers' terms (in particular CC BY-NC 4.0 for PU) apply to the data, not to the code, which is MIT licensed (LICENSE).
The prediction files contain scores, labels and identifiers only; those derived from PU are shared for noncommercial academic use under the provider's terms, which the reader should check before any other use.  Split lists, features and saved logits of the third-party benchmark are not included.
