#ifndef MEGATRACE_EXTRA_FIELD_RESOLVER_H
#define MEGATRACE_EXTRA_FIELD_RESOLVER_H

#include <string>

namespace megatrace {

/**
 * Context passed to extra field resolvers.
 * Extend this struct when adding resolvers that need more context.
 */
struct ExtraFieldContext {
    const std::string& pod_name;
    int save_iter;
};

/**
 * Resolver type: (context) -> value string.
 * Return empty string to skip output for this field.
 */
using ExtraFieldResolver = std::string (*)(const ExtraFieldContext&);

/**
 * Register a built-in computed field resolver.
 * Call before log_writer_thread runs (e.g. at start of log_writer_thread).
 * If name is already registered, the new resolver replaces the old one.
 */
void register_extra_field_resolver(const std::string& name, ExtraFieldResolver resolver);

/**
 * Get value for field name: if registered as built-in, call resolver; else use getenv(name).
 * Returns value string; empty means skip output for this field.
 */
std::string get_extra_field_value(const std::string& name, const ExtraFieldContext& ctx);

/**
 * Initialize built-in resolvers (RUNNING_ROUND, etc.).
 * Called internally by log_writer_thread; external code may call to ensure built-ins exist before adding custom resolvers.
 */
void init_builtin_extra_resolvers();

}  // namespace megatrace

#endif /* MEGATRACE_EXTRA_FIELD_RESOLVER_H */
