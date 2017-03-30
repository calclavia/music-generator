# Mount AWS volume
# http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-using-volumes.html
# TODO: Add auto-remount

lsblk # List volumes
sudo mkdir /silo # Create volume path
sudo mount /dev/xvdf /silo # Mount the volume
