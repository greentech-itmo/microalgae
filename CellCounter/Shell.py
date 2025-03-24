

def calculate_cells_count(N: int, D: int):
    """
    Function for calculation cells count for standard Goryaev chamber
    height=0.1mm, small square size=0.05mm*0.05mm
    In our experiments photo image size from electron microscope (0.17085mm*0.12814mm)
    Volume of area from photo=0.002189mm^2

    :param N: number of cells on photo
    :param D: dilution factor
    """
    cells_count = (N*D)/0.002189
    return  cells_count
