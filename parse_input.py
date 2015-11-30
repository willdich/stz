from ConfigParser import ConfigParser

# Parse configuration file
def parse_input(config_file):

    config = ConfigParser()
    config.read(config_file)

    # get params from file, using test values as defaults if missing options
    lambd = config.getfloat('Material Parameters', 'lambd') if config.has_option('Material Parameters', 'lambd') else .5
    mu = config.getfloat('Material Parameters', 'mu') if config.has_option('Material Parameters', 'mu') else 1.
    rho = config.getfloat('Material Parameters', 'rho') if config.has_option('Material Parameters', 'rho') else 3.

    min_x = config.getfloat('Coordinates', 'min_x') if config.has_option('Coordinates', 'min_x') else 0.
    max_x = config.getfloat('Coordinates', 'max_x') if config.has_option('Coordinates', 'max_x') else 5.
    min_y = config.getfloat('Coordinates', 'min_y') if config.has_option('Coordinates', 'min_y') else 0.
    max_y = config.getfloat('Coordinates', 'max_y') if config.has_option('Coordinates', 'max_y') else 5.
    min_z = config.getfloat('Coordinates', 'min_z') if config.has_option('Coordinates', 'min_z') else 0.
    max_z = config.getfloat('Coordinates', 'max_z') if config.has_option('Coordinates', 'max_z') else 5.

    N_x = config.getint('Grid Points', 'N_x') if config.has_option('Grid Points', 'N_x') else 100
    N_y = config.getint('Grid Points', 'N_y') if config.has_option('Grid Points', 'N_y') else 100
    N_z = config.getint('Grid Points', 'N_z') if config.has_option('Grid Points', 'N_z') else 100

    t_0 = config.getfloat('Time Parameters', 't_0') if config.has_option('Time Parameters', 't_0') else 0.
    t_f = config.getfloat('Time Parameters', 't_f') if config.has_option('Time Parameters', 't_f') else 2.5
    N_t = config.getint('Time Parameters', 'N_t') if config.has_option('Time Parameters', 'N_t') else 100

    return lambd, mu, rho, min_x, max_x, min_y, max_y, min_z, max_z, N_x, N_y, N_z, t_0, t_f, N_t
