CREATE TABLE velib_status (
    id SERIAL PRIMARY KEY, 
    station varchar, 
    bikes integer, 
    max_bikes integer, 
    mechanical integer, 
    ebike integer, 
    is_installed integer, 
    is_returning integer, 
    is_renting integer, 
    dt timestamp, 
    delta integer, 
    poll_dt timestamp
    );
