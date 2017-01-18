#include <QApplication>
#include <iostream>
#include "Viewer.h"
#include <QDebug>

int main(int argc, char * argv[]) {
	if (argc != 2) {
		qDebug() << "usage: viewer <port number>";
		return 1;
	}
	QApplication a(argc, argv);
	unsigned short clientPort = (unsigned short)atoi(argv[1]);
	Viewer viewer(clientPort);
	viewer.resize(512, 512);
	viewer.create();
	viewer.show();

	return a.exec();
}
