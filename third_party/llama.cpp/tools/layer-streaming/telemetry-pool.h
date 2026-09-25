#pragma once

#include <memory>
#include <string>
#include <vector>

class ls_pool_validator {
public:
    ls_pool_validator();
    ~ls_pool_validator();

    ls_pool_validator(const ls_pool_validator &) = delete;
    ls_pool_validator & operator=(const ls_pool_validator &) = delete;

    std::vector<std::string> init_errors(const std::string & line);
    std::vector<std::string> byte_account_errors(const std::string & line);
    std::vector<std::string> run_start_errors(const std::string & line);
    std::vector<std::string> event_errors(const std::string & line);
    std::vector<std::string> fetch_errors(const std::string & line,
                                           long long layer, long long generation,
                                           long long ubatch, long long route_union,
                                           long long fetched, long long resident,
                                           long long retained, long long bytes_staged,
                                           long long bytes_full_bank);
    std::vector<std::string> final_errors(bool refusal) const;

    bool enabled() const;
    void set_n_expert(long long n_expert);

private:
    struct model;
    std::unique_ptr<model> state_;
};
