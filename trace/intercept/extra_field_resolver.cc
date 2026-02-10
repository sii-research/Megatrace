#include "extra_field_resolver.h"
#include <stdlib.h>
#include <cstring>
#include <algorithm>
#include <map>

namespace megatrace {

static std::map<std::string, ExtraFieldResolver>* g_resolvers = nullptr;

static std::map<std::string, ExtraFieldResolver>& get_resolvers() {
    if (g_resolvers == nullptr) {
        g_resolvers = new std::map<std::string, ExtraFieldResolver>();
    }
    return *g_resolvers;
}

void register_extra_field_resolver(const std::string& name, ExtraFieldResolver resolver) {
    get_resolvers()[name] = resolver;
}

static std::string resolve_running_round(const ExtraFieldContext& ctx) {
    const std::string& pod_name = ctx.pod_name;
    int dash_count = std::count(pod_name.begin(), pod_name.end(), '-');
    // example pod name job-34e2a67a-d3d5-43ef-9d3a-07e9986a7355-worker-1-8
    if (dash_count < 8) {
        return "0";
    }
    size_t last_dash_pos = pod_name.find_last_of('-');
    if (last_dash_pos == std::string::npos) {
        return "0";
    }
    return pod_name.substr(last_dash_pos + 1);
}

void init_builtin_extra_resolvers() {
    static bool initialized = false;
    if (initialized) return;
    initialized = true;
    register_extra_field_resolver("RUNNING_ROUND", resolve_running_round);
}

std::string get_extra_field_value(const std::string& name, const ExtraFieldContext& ctx) {
    auto& resolvers = get_resolvers();
    auto it = resolvers.find(name);
    if (it != resolvers.end()) {
        return it->second(ctx);
    }
    const char* val = getenv(name.c_str());
    return (val && *val != '\0') ? std::string(val) : std::string();
}

}  // namespace megatrace
