#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <errno.h>

void print_error(const char *msg)
{
    perror(msg);
    exit(EXIT_FAILURE);
}

int main()
{
    int sockfd;

    FILE *fp = fopen("socket_options.txt", "w");
    if (fp == NULL)
    {
        perror("File open error");
        exit(EXIT_FAILURE);
    }

    // Create TCP socket
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0)
    {
        print_error("Socket creation failed");
    }

    /*------------------ IP Layer Options ------------------*/

    int ip_ttl;
    socklen_t optlen = sizeof(ip_ttl);

    if (getsockopt(sockfd, IPPROTO_IP, IP_TTL, &ip_ttl, &optlen) < 0)
        print_error("getsockopt IP_TTL");

    fprintf(fp, "IP_TTL          : %d\n", ip_ttl);

    int ip_hdrincl;
    optlen = sizeof(ip_hdrincl);

    if (getsockopt(sockfd, IPPROTO_IP, IP_HDRINCL,
                   &ip_hdrincl, &optlen) < 0)
    {
        fprintf(fp,
                "IP_HDRINCL      : Not applicable to TCP (%s)\n",
                strerror(errno));
    }
    else
    {
        fprintf(fp, "IP_HDRINCL      : %d\n", ip_hdrincl);
    }

    /*------------------ TCP Layer Options ------------------*/

    int tcp_nodelay;
    optlen = sizeof(tcp_nodelay);

    if (getsockopt(sockfd, IPPROTO_TCP,
                   TCP_NODELAY,
                   &tcp_nodelay,
                   &optlen) < 0)
    {
        print_error("getsockopt TCP_NODELAY");
    }

    fprintf(fp, "TCP_NODELAY     : %d\n", tcp_nodelay);

    int tcp_maxseg;
    optlen = sizeof(tcp_maxseg);

    if (getsockopt(sockfd, IPPROTO_TCP,
                   TCP_MAXSEG,
                   &tcp_maxseg,
                   &optlen) < 0)
    {
        print_error("getsockopt TCP_MAXSEG");
    }

    fprintf(fp, "TCP_MAXSEG      : %d\n", tcp_maxseg);

#ifdef TCP_CORK
    int tcp_cork;
    optlen = sizeof(tcp_cork);

    if (getsockopt(sockfd,
                   IPPROTO_TCP,
                   TCP_CORK,
                   &tcp_cork,
                   &optlen) == 0)
    {
        fprintf(fp, "TCP_CORK        : %d\n", tcp_cork);
    }
#endif

#ifdef TCP_KEEPIDLE
    int keep_idle;
    optlen = sizeof(keep_idle);

    if (getsockopt(sockfd,
                   IPPROTO_TCP,
                   TCP_KEEPIDLE,
                   &keep_idle,
                   &optlen) == 0)
    {
        fprintf(fp, "TCP_KEEPIDLE    : %d seconds\n", keep_idle);
    }
#endif

#ifdef TCP_KEEPINTVL
    int keep_intvl;
    optlen = sizeof(keep_intvl);

    if (getsockopt(sockfd,
                   IPPROTO_TCP,
                   TCP_KEEPINTVL,
                   &keep_intvl,
                   &optlen) == 0)
    {
        fprintf(fp, "TCP_KEEPINTVL   : %d seconds\n", keep_intvl);
    }
#endif

#ifdef TCP_KEEPCNT
    int keep_cnt;
    optlen = sizeof(keep_cnt);

    if (getsockopt(sockfd,
                   IPPROTO_TCP,
                   TCP_KEEPCNT,
                   &keep_cnt,
                   &optlen) == 0)
    {
        fprintf(fp, "TCP_KEEPCNT     : %d probes\n", keep_cnt);
    }
#endif

    fclose(fp);
    close(sockfd);

    printf("Socket options written to socket_options.txt\n");

    return 0;
}

// gcc cn3.c -o cn3