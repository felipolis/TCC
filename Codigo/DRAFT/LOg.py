import logging

# Configuração do logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Cria um manipulador de arquivo para escrever logs em um arquivo separado
file_handler = logging.FileHandler('app.log')

# Define o nível de logging para o manipulador de arquivo
file_handler.setLevel(logging.DEBUG)

# Define o formato das mensagens de log no arquivo
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Adiciona o manipulador de arquivo ao logger
logger.addHandler(file_handler)

# Exemplos de logging
logger.debug('Esta é uma mensagem de debug')
logger.info('Esta é uma mensagem de informação')
logger.warning('Esta é uma mensagem de aviso')
logger.error('Esta é uma mensagem de erro')
logger.critical('Esta é uma mensagem crítica')
