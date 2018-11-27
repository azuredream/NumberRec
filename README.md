# NumberRec
##For User
  input"docker stack deploy -c docker-compose.yml webcube"
  to start the project.
  URL:localhost

##For Developer
  app.py:      Flask view file.
  form:        The CNN Model.
  model.py:    Model Trainer.
  mnistout.py: Export MNIST data for test
  recpic.py:   Test Model reading function from form
  templates:   Htmls
  static:      JS,CSS
  data:        Cassandra and redis will save data in this dir
  Dockerfile,requirements.txt :   Container Init (Docker config)
  docker-compose.yml:             Config docker services (Docker config)
