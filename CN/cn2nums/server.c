#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <netinet/in.h>

#define PORT 8080

int main() {
    int server_fd, new_socket;
    struct sockaddr_in address;
    int addrlen = sizeof(address);

    int buffer[3];
    int result[3];

    // Create socket
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("Socket failed");
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    // Bind socket
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("Bind failed");
        exit(EXIT_FAILURE);
    }

    // Listen
    if (listen(server_fd, 3) < 0) {
        perror("Listen failed");
        exit(EXIT_FAILURE);
    }

    printf("Server listening on port %d...\n", PORT);

    // Accept client
    new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t *)&addrlen);
    if (new_socket < 0) {
        perror("Accept failed");
        exit(EXIT_FAILURE);
    }

    // Receive numbers
    read(new_socket, buffer, sizeof(buffer));

    int a = buffer[0];
    int b = buffer[1];
    int c = buffer[2];

    result[0] = a + b + c;
    result[1] = a - b - c;
    result[2] = a * b * c;

    // Send results
    write(new_socket, result, sizeof(result));

    printf("Numbers received: %d %d %d\n", a, b, c);
    printf("Results sent successfully.\n");

    close(new_socket);
    close(server_fd);

    return 0;
}
// gcc server.c -o server
// ./server