# circuit_coords.py


class CircuitCoords:
    """
    Official geographic coordinates of the circuits (lat, lon)
    """

    CIRCUIT_COORDS: dict[str, tuple[float, float]] = {
        "Bahrain": (26.0325, 50.5106),
        "Saudi Arabia": (21.6319, 39.1044),
        "Australia": (-37.8497, 144.9680),
        "Japan": (34.8431, 136.5407),
        "China": (31.3389, 121.2200),
        "Miami": (25.9581, -80.2389),
        "Emilia Romagna": (44.3439, 11.7167),
        "Monaco": (43.7347, 7.4206),
        "Canada": (45.5000, -73.5228),
        "Spain": (41.5700, 2.2611),
        "Austria": (47.2197, 14.7647),
        "Great Britain": (52.0786, -1.0169),
        "Hungary": (47.5789, 19.2486),
        "Belgium": (50.4372, 5.9714),
        "Netherlands": (52.3888, 4.5409),
        "Italy": (45.6156, 9.2811),
        "Azerbaijan": (40.3725, 49.8533),
        "Singapore": (1.2914, 103.8640),
        "United States": (30.1328, -97.6411),
        "Mexico": (19.4042, -99.0907),
        "São Paulo": (-23.7036, -46.6997),
        "Las Vegas": (36.1147, -115.1728),
        "Qatar": (25.4900, 51.4542),
        "Abu Dhabi": (24.4672, 54.6031),
    }

    @classmethod
    def get_coords(cls, circuit: str) -> tuple[float, float]:
        """
        Returns coordinates (lat, lon). Default = (0,0) if it does not exist.
        """
        return cls.CIRCUIT_COORDS.get(circuit, (0.0, 0.0))