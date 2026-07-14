#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <netinet/in.h>

#define PORT 9090      // Port number

int main() {
    int server_fd, new_socket;
    struct sockaddr_in address;
    int addrlen = sizeof(address);
    char buffer[128];

    // Create socket
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == 0) {
        perror("Socket failed");
        exit(EXIT_FAILURE);
    }

    // Configure server address
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    // Bind socket
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("Bind failed");
        exit(EXIT_FAILURE);
    }

    // Listen for connections
    if (listen(server_fd, 3) < 0) {
        perror("Listen failed");
        exit(EXIT_FAILURE);
    }

    printf("Daytime server listening on port %d...\n", PORT);

    while (1) {

        // Accept client connection
        new_socket = accept(server_fd,
                           (struct sockaddr *)&address,
                           (socklen_t *)&addrlen);

        if (new_socket < 0) {
            perror("Accept failed");
            continue;
        }

        // Get current system time
        time_t now = time(NULL);
        struct tm *tm_info = localtime(&now);

        // Format date and time
        strftime(buffer, sizeof(buffer),
                 "%Y-%m-%d %H:%M:%S\n",
                 tm_info);

        // Send time to client
        write(new_socket, buffer, strlen(buffer));

        // Close client socket
        close(new_socket);
    }

    close(server_fd);
    return 0;
}

//  gcc server.c -o server