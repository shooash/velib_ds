SELECT a.id AS status_id, a.station, a.bikes, EXTRACT(hour FROM a.dt) AS hour, EXTRACT(isodow FROM a.dt) AS weekday, b.capacity, b.lat, b.lon, b.name, a.dt, a.delta, a.poll_dt 
FROM velib_status a 
JOIN velib_stations b ON a.station::text = b.station::text;
