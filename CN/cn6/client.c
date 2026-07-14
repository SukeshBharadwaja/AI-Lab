#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#define PORT 8080
#define BUFFER_SIZE 1024

int main() {
    int sock;
    struct sockaddr_in serv_addr;

    char message[] = "Hello from Sukesh!";
    char buffer[BUFFER_SIZE] = {0};

    // Create socket
    sock = socket(AF_INET, SOCK_STREAM, 0);

    if (sock < 0) {
        perror("Socket creation failed");
        return -1;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);

    if (inet_pton(AF_INET, "127.0.0.1",
                  &serv_addr.sin_addr) <= 0) {
        perror("Invalid address");
        return -1;
    }

    // Connect
    if (connect(sock,
                (struct sockaddr *)&serv_addr,
                sizeof(serv_addr)) < 0) {
        perror("Connection failed");
        return -1;
    }

    // Send
    send(sock, message, strlen(message), 0);

    // Receive echo
    int bytes = read(sock, buffer, BUFFER_SIZE - 1);

    if (bytes > 0) {
        buffer[bytes] = '\0';
        printf("Echo from server: %s\n", buffer);
    }

    close(sock);

    return 0;
}