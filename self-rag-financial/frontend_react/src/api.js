export const API = {
    baseUrl: () => localStorage.getItem('selfrag_api_url') || 'http://localhost:8000',
    apiKey: () => localStorage.getItem('selfrag_api_key') || '',

    headers: () => ({
        'Content-Type': 'application/json',
        'X-API-Key': API.apiKey()
    }),

    async request(method, path, body = null) {
        const opts = { method, headers: API.headers() };
        if (body) opts.body = JSON.stringify(body);
        
        try {
            const resp = await fetch(API.baseUrl() + path, opts);
            if (!resp.ok) {
                let errStr = `HTTP ${resp.status}`;
                try {
                    const err = await resp.json();
                    errStr = err.detail || errStr;
                } catch(e){}
                throw new Error(errStr);
            }
            return await resp.json();
        } catch (error) {
            throw error;
        }
    },

    async upload(path, formData) {
        try {
            const resp = await fetch(API.baseUrl() + path, {
                method: 'POST',
                headers: { 'X-API-Key': API.apiKey() },
                body: formData
            });
            if (!resp.ok) {
                let errStr = `HTTP ${resp.status}`;
                try {
                    const err = await resp.json();
                    errStr = err.detail || errStr;
                } catch(e){}
                throw new Error(errStr);
            }
            return await resp.json();
        } catch (error) {
            throw error;
        }
    },

    health: () => API.request('GET', '/health'),
    query: (body) => API.request('POST', '/query', body),
    sessions: async (archived=false) => {
        let res = await API.request('GET', `/sessions?include_archived=${archived}`);
        if (res && !Array.isArray(res)) {
            if (Array.isArray(res.sessions)) return res.sessions;
            return Object.values(res);
        }
        return res;
    },
    getSession: (id) => API.request('GET', `/sessions/${id}`),
    createSession: (body) => API.request('POST', '/sessions', body),
    archiveSession: (id) => API.request('DELETE', `/sessions/${id}?archive=true`),
    docsInfo: () => API.request('GET', '/docs-info'),
    metrics: () => API.request('GET', '/metrics'),
    cacheStats: () => API.request('GET', '/cache/stats'),
    ingest: (formData) => API.upload('/ingest', formData)
};
