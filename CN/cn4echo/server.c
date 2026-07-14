#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#define PORT 8080
#define BUF_SIZE 1024

int main()
{
    int server_fd, new_socket;
    struct sockaddr_in address;
    int addrlen = sizeof(address);
    char buffer[BUF_SIZE] = {0};

    // Create socket
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0)
    {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    // Allow port reuse
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    // Configure address
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    // Bind
    if (bind(server_fd, (struct sockaddr *)&address,
             sizeof(address)) < 0)
    {
        perror("Bind failed");
        exit(EXIT_FAILURE);
    }

    // Listen
    if (listen(server_fd, 3) < 0)
    {
        perror("Listen failed");
        exit(EXIT_FAILURE);
    }

    printf("TCP Echo Server listening on port %d...\n", PORT);

    // Accept client
    new_socket = accept(server_fd,
                        (struct sockaddr *)&address,
                        (socklen_t *)&addrlen);

    if (new_socket < 0)
    {
        perror("Accept failed");
        exit(EXIT_FAILURE);
    }

    // Receive message
    int valread = read(new_socket, buffer, BUF_SIZE);

    if (valread > 0)
    {
        buffer[valread] = '\0';
        printf("Received from client: %s\n", buffer);

        // Echo back
        send(new_socket, buffer, strlen(buffer), 0);

        printf("Echo sent successfully.\n");
    }

    close(new_socket);
    close(server_fd);

    return 0;
}