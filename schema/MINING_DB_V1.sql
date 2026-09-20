PRAGMA foreign_keys = ON;

CREATE TABLE schema_meta (
    key TEXT PRIMARY KEY NOT NULL,
    value TEXT NOT NULL
);

INSERT INTO schema_meta(key, value) VALUES
    ('schema_name','MINING_DB_V1'),
    ('schema_version','1'),
    ('move_root_sentinel','ROOT');

CREATE TABLE engines (
    engine_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    stockfish_commit TEXT,
    executable_path TEXT,
    created_utc TEXT
);

CREATE TABLE networks (
    network_id INTEGER PRIMARY KEY,
    role_hint TEXT CHECK (
        role_hint IS NULL OR
        role_hint IN ('official','candidate','teacher','other')
    ),
    name TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    path TEXT,
    description TEXT,
    created_utc TEXT
);

CREATE TABLE positions (
    position_id INTEGER PRIMARY KEY,
    fen TEXT NOT NULL UNIQUE,
    fen_sha256 TEXT NOT NULL UNIQUE,
    position_key TEXT NOT NULL,
    side_to_move TEXT NOT NULL CHECK (side_to_move IN ('w','b')),
    halfmove_clock INTEGER NOT NULL CHECK (halfmove_clock >= 0),
    fullmove_number INTEGER NOT NULL CHECK (fullmove_number >= 1),
    source TEXT,
    split TEXT CHECK (
        split IS NULL OR
        split IN ('train','val','challenge','fresh','anchor','other')
    ),
    original_score_type TEXT CHECK (
        original_score_type IS NULL OR
        original_score_type IN ('cp','mate')
    ),
    original_score_value INTEGER,
    original_wdl_w INTEGER,
    original_wdl_d INTEGER,
    original_wdl_l INTEGER,
    created_utc TEXT
);

CREATE TABLE mining_runs (
    run_id INTEGER PRIMARY KEY,
    run_tag TEXT NOT NULL UNIQUE,
    created_utc TEXT NOT NULL,
    engine_id INTEGER NOT NULL,
    candidate_network_id INTEGER NOT NULL,
    official_network_id INTEGER NOT NULL,
    teacher_network_id INTEGER NOT NULL,
    candidate_nodes INTEGER,
    candidate_multipv INTEGER,
    teacher_nodes INTEGER,
    threads INTEGER,
    hash_mb INTEGER,
    uci_show_wdl INTEGER NOT NULL DEFAULT 1
        CHECK (uci_show_wdl IN (0,1)),
    description TEXT,
    FOREIGN KEY(engine_id) REFERENCES engines(engine_id),
    FOREIGN KEY(candidate_network_id) REFERENCES networks(network_id),
    FOREIGN KEY(official_network_id) REFERENCES networks(network_id),
    FOREIGN KEY(teacher_network_id) REFERENCES networks(network_id)
);

CREATE TABLE search_jobs (
    job_id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL,
    position_id INTEGER NOT NULL,
    engine_role TEXT NOT NULL CHECK (
        engine_role IN ('candidate','official','teacher')
    ),
    engine_id INTEGER NOT NULL,
    network_id INTEGER NOT NULL,
    search_kind TEXT NOT NULL CHECK (
        search_kind IN ('multipv','searchmove','singlepv','verification')
    ),
    restricted_move TEXT NOT NULL DEFAULT 'ALL',
    requested_nodes INTEGER CHECK (
        requested_nodes IS NULL OR requested_nodes > 0
    ),
    requested_multipv INTEGER NOT NULL DEFAULT 1
        CHECK (requested_multipv >= 1),
    status TEXT NOT NULL DEFAULT 'pending' CHECK (
        status IN ('pending','running','ok','no_legal_moves','error')
    ),
    bestmove TEXT,
    ponder TEXT,
    error_text TEXT,
    started_utc TEXT,
    finished_utc TEXT,

    UNIQUE (
        run_id,
        position_id,
        engine_role,
        search_kind,
        restricted_move
    ),

    FOREIGN KEY(run_id) REFERENCES mining_runs(run_id),
    FOREIGN KEY(position_id) REFERENCES positions(position_id),
    FOREIGN KEY(engine_id) REFERENCES engines(engine_id),
    FOREIGN KEY(network_id) REFERENCES networks(network_id)
);

CREATE TABLE search_results (
    job_id INTEGER NOT NULL,
    rank INTEGER NOT NULL CHECK (rank >= 1),
    move TEXT NOT NULL,
    score_type TEXT NOT NULL CHECK (score_type IN ('cp','mate')),
    score_value INTEGER,
    bound TEXT NOT NULL DEFAULT 'exact'
        CHECK (bound IN ('exact','lowerbound','upperbound')),
    wdl_w INTEGER,
    wdl_d INTEGER,
    wdl_l INTEGER,
    depth INTEGER,
    seldepth INTEGER,
    nodes INTEGER,
    time_ms INTEGER,
    nps INTEGER,
    pv TEXT,

    PRIMARY KEY (job_id, rank),

    FOREIGN KEY(job_id)
        REFERENCES search_jobs(job_id)
        ON DELETE CASCADE
);

CREATE TABLE static_evals (
    static_eval_id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL,
    position_id INTEGER NOT NULL,
    network_role TEXT NOT NULL CHECK (
        network_role IN ('candidate','official','teacher')
    ),
    network_id INTEGER NOT NULL,

    move TEXT NOT NULL,

    child_fen TEXT,

    status TEXT NOT NULL DEFAULT 'ok' CHECK (
        status IN (
            'ok',
            'in_check',
            'terminal',
            'illegal_move',
            'parse_error',
            'engine_error'
        )
    ),

    score_raw_stm INTEGER,
    score_root_pov INTEGER,

    error_text TEXT,

    UNIQUE (
        run_id,
        position_id,
        network_role,
        move
    ),

    CHECK (
        (
            status = 'ok'
            AND score_raw_stm IS NOT NULL
            AND score_root_pov IS NOT NULL
        )
        OR status <> 'ok'
    ),

    FOREIGN KEY(run_id) REFERENCES mining_runs(run_id),
    FOREIGN KEY(position_id) REFERENCES positions(position_id),
    FOREIGN KEY(network_id) REFERENCES networks(network_id)
);

CREATE TABLE mining_metrics (
    run_id INTEGER NOT NULL,
    position_id INTEGER NOT NULL,

    candidate_bestmove TEXT,
    teacher_bestmove TEXT,

    candidate_margin_internal REAL,

    teacher_regret_internal REAL,
    teacher_regret_wdl_expected REAL,

    candidate_static_preference REAL,
    official_static_preference REAL,
    deep_teacher_preference REAL,

    static_disagreement REAL,
    search_amplification REAL,

    rank_error REAL,
    topk_overlap REAL,

    structural_persistence REAL,
    break_score REAL,

    primary_class TEXT,
    metrics_version TEXT NOT NULL DEFAULT 'V1',

    PRIMARY KEY (run_id, position_id),

    FOREIGN KEY(run_id)
        REFERENCES mining_runs(run_id)
        ON DELETE CASCADE,

    FOREIGN KEY(position_id)
        REFERENCES positions(position_id)
);

CREATE TABLE position_classes (
    run_id INTEGER NOT NULL,
    position_id INTEGER NOT NULL,
    class_name TEXT NOT NULL,
    class_score REAL,
    classifier_version TEXT NOT NULL DEFAULT 'V1',

    PRIMARY KEY (
        run_id,
        position_id,
        class_name
    ),

    FOREIGN KEY(run_id)
        REFERENCES mining_runs(run_id)
        ON DELETE CASCADE,

    FOREIGN KEY(position_id)
        REFERENCES positions(position_id)
);

CREATE INDEX idx_positions_position_key
    ON positions(position_key);

CREATE INDEX idx_positions_source_split
    ON positions(source, split);

CREATE INDEX idx_search_jobs_run_position
    ON search_jobs(run_id, position_id);

CREATE INDEX idx_search_jobs_status
    ON search_jobs(run_id, status);

CREATE INDEX idx_search_results_move
    ON search_results(move);

CREATE INDEX idx_static_evals_run_position
    ON static_evals(run_id, position_id);

CREATE INDEX idx_static_evals_move
    ON static_evals(move);

CREATE INDEX idx_metrics_break_score
    ON mining_metrics(run_id, break_score DESC);

CREATE INDEX idx_metrics_regret
    ON mining_metrics(run_id, teacher_regret_wdl_expected DESC);

CREATE INDEX idx_metrics_class
    ON mining_metrics(run_id, primary_class);

CREATE INDEX idx_classes_lookup
    ON position_classes(run_id, class_name, class_score DESC);
