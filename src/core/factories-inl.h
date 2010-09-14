#ifndef FACTORIES_INL_H_
#error "Direct inclusion of this file is not allowed, include factories.h"
#endif
#undef FACTORIES_INL_H_

namespace mll {

template<typename TFactory, typename TObject>
TFactory& FactoryBase<TFactory, TObject>::Instance() {
    static TFactory factory;
    return factory;
}

template<typename TFactory, typename TObject>
bool FactoryBase<TFactory, TObject>::Register(Entry entry) {
    if (entry.IsEmpty() ||
        entry.GetName().empty() ||
        entry.GetDescription().empty())
    {
        return false;
    }
    std::string name = entry.GetName();
    ToLower(&name);
    return entries_.insert(std::make_pair(name, entry)).second;
}

template<typename TFactory, typename TObject>
bool FactoryBase<TFactory, TObject>::Contains(const std::string& name) const {
    return !GetEntry(name).IsEmpty();
}


template<typename TFactory, typename TObject>
const typename FactoryBase<TFactory, TObject>::Entry& FactoryBase<TFactory, TObject>::GetEntry(
    const std::string& name) const
{
    std::string localName = name;
    ToLower(&localName);
    typename std::map<std::string, Entry>::const_iterator mapIt = entries_.find(localName);
    if (mapIt == entries_.end()) {
        return Entry::EmptyEntry();
    }
    return mapIt->second;
}

template<typename TFactory, typename TObject>
sh_ptr<TObject> FactoryBase<TFactory, TObject>::Create(const std::string& name) const {
    const Entry& entry = GetEntry(name);
    return entry.IsEmpty() ? sh_ptr<TObject>(NULL) : entry.GetObject().Clone();
}

template<typename TFactory, typename TObject>
void FactoryBase<TFactory, TObject>::GetEntries(std::vector< Entry >* entries) const {
    for (typename std::map<std::string, Entry>::const_iterator mapIt = entries_.begin();
         mapIt != entries_.end();
         ++mapIt) {
        entries->push_back(mapIt->second);
    }
}

} // namespace mll
