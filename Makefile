.PHONY: all update_theme install_dependencies serve

update_theme:
	hugo mod get -u
	hugo mod tidy

install:
	hugo mod tidy

serve:
	hugo server --logLevel debug --disableFastRender -p 1313
