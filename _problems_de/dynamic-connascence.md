---
title: Dynamic Connascence
description: Ein Name, Dateipfad oder String-Literal kodiert eine Laufzeit-Referenz
  zwischen Erzeuger und Verbraucher, die der Compiler sowie automatisiertes
  Umbenennen oder Verschieben beim Refactoring nicht erkennen können.
category:
- Architecture
- Code
related_problems:
- slug: poor-naming-conventions
  similarity: 0.5
- slug: hidden-dependencies
  similarity: 0.5
solutions:
- dependency-injection
- dependency-injection-container
- static-analysis-and-linting
- characterization-tests
- change-impact-analysis
- consumer-driven-contracts
- typed-schema-extraction
- explicit-extension-points
- automated-code-migration
- schema-registry
layout: problem
lang: de
en_slug: dynamic-connascence
---

## Description

Dynamische Connascence liegt vor, wenn zwei Teile eines Systems sich auf einen Namen einigen müssen, diese Übereinkunft aber nur zur Laufzeit durchgesetzt wird statt vom Compiler oder der statischen Analyse einer IDE. Ein reflektiver Klassen-Lookup, der aus einem berechneten String aufgebaut wird, eine UI-Komponente, die über einen identischen Dateipfad mit ihrem Template gekoppelt ist, oder ein persistiertes JSON-Feld, das ein Deserialisierer unter einem exakten Schlüssel erwartet, sind allesamt Beispiele dafür. Weil die Referenz nur als String oder Namenskonvention existiert, rutscht eine gewöhnliche Symbol-Umbenennung oder Dateiverschiebung unbemerkt hindurch: automatisierte Refactoring-Werkzeuge, Compiler und einfache Textsuche erkennen sie alle nicht als dieselbe Referenz. Der Fehler zeigt sich dann weit entfernt von der eigentlichen Änderung, oft erst wenn der betroffene Codepfad tatsächlich ausgeführt wird, und häufig als stiller Fallback oder verwirrender Laufzeitfehler statt als Build-Fehler. Das Konzept stammt aus Meilir Page-Jones' Connascence-Taxonomie, die statische Connascence, erkennbar durch Lesen des Quellcodes, von dynamischer Connascence unterscheidet, die nur durch Ausführen des Programms erkennbar ist.

## Indicators ⟡

- Ein reflektiver oder dynamischer Lookup (`Class.forName`, `loadClass`, dynamisches `import()`, stringbasierter Service-Lookup) baut einen Namen aus einem String zusammen, der anderswo erzeugt oder konfiguriert wird.
- Zwei Dateien sind nur deshalb gepaart, weil sie denselben relativen Pfad und Basisnamen teilen, ohne explizite Referenz zwischeneinander.
- Das Grep nach einem Klassen-, Feld- oder Dateinamen übersieht reale Verwendungen, weil manche zur Laufzeit durch Präfixe, Suffixe oder Konkatenation zusammengesetzt werden.
- Eine Umbenennung, die die „Find Usages“-Funktion einer IDE oder ein automatisiertes Refactoring als vollständig abgedeckt meldete, bricht nach dem Deployment dennoch etwas.
- Persistierte Daten wie Datenbankzeilen, Konfigurationsdateien oder exportierte Lesezeichen enthalten Klassennamen, Schlüssel oder IDs, die exakt zum Code passen müssen.

## Symptoms ▲

- [Versteckte Abhängigkeiten](versteckte-abhaengigkeiten.md)
<br/>  Die Verbindung zwischen Erzeuger und Verbraucher existiert nur als gemeinsamer String, sodass die Abhängigkeit in den Schnittstellen oder der Struktur des Codes unsichtbar bleibt.
- [Regressionsfehler](regressionsfehler.md)
<br/>  Eine Umbenennung oder Verschiebung, die jede automatisierte Prüfung besteht, bricht dennoch den Laufzeit-Lookup und erzeugt eine Regression, die keine Testsuite vorhergesehen hat.
- [Debugging-Schwierigkeiten](debugging-schwierigkeiten.md)
<br/>  Wenn sich der Fehler zeigt, gibt es keinen Stacktrace, der zur Umbenennung zurückführt; Entwickler müssen den Namensvertrag von Hand rekonstruieren.
- [Angst vor Breaking Changes](angst-vor-breaking-changes.md)
<br/>  Wurde ein Team einmal von einem unsichtbaren Namensvertrag überrascht, scheut es sich, irgendetwas in der Nähe von reflektivem oder konventionsbasiertem Code umzubenennen oder zu verschieben.

## Causes ▼

- [Stringly Typed Code](stringly-typed-code.md)
<br/>  Fachliche Werte, die als rohe Strings dargestellt werden, sind genau das, woraus dynamische Lookups, Präfixe und Query-Pfade gebaut werden — das macht aus einem Wertproblem ein Kopplungsproblem.
- [Termindruck](termindruck.md)
<br/>  Reflexions- oder konventionsbasierte Verdrahtung ist schneller geschrieben als explizite, typisierte Verdrahtung, sodass unter Druck stehende Entwickler danach greifen, ohne das Refactoring-Risiko abzuwägen.
- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Entwickler, die mit der Convention-over-Configuration-Magie eines Frameworks nicht vertraut sind, erkennen gar nicht, dass sie dabei einen impliziten Laufzeitvertrag schaffen.

## Detection Methods ○

- **Statische Mustersuche:** Durchsuchen Sie die Codebasis per Grep nach reflektiven oder dynamischen Lade-APIs (`Class.forName`, `loadClass`, dynamisches `import()`) und String-Konkatenation unmittelbar davor.
- **Audit struktureller Paarungen:** Vergleichen Sie Quell- und Ressourcenbäume auf Dateien, die nur durch identische relative Pfade und Basisnamen verbunden sind.
- **Scan nach Konstantennamen:** Suchen Sie nach Konstanten mit Namen wie `PREFIX`, `SUFFIX`, `SEPARATOR`, `SERVICE_NAME` oder ähnlichem, die auf ein selbstgebautes Namensprotokoll hindeuten.
- **Umbenennungs-Probelauf:** Führen Sie vor einer echten Umbenennung einen Testlauf in einem Branch durch und lassen Sie die vollständige Test- und Integrationssuite laufen, nicht nur den Compiler, um zu sehen, was nur zur Laufzeit bricht.
- **Stichprobe persistierter Daten:** Prüfen Sie Produktionsdaten, Konfiguration und Exporte auf gespeicherte Klassennamen, Schlüssel oder IDs, die eine Umbenennung stillschweigend ungültig machen würde.

## Examples

Ein CMS löst die Listenansicht eines Domänentyps auf, indem es `type.getSimpleName()` nimmt, `"ListPage"` anhängt und diese Klasse reflektiv lädt, um die Admin-Oberfläche zu rendern. Ein Entwickler benennt das Modell `Invoice` mit der automatisierten Umbenennung seiner IDE in `Bill` um, die brav jede statisch typisierte Referenz aktualisiert; der Build ist grün, die Tests laufen durch. In Produktion wirft die Admin-Listenansicht für Rechnungen jedoch eine `ClassNotFoundException`, weil der reflektive Lookup nun nach `BillListPage` sucht — einer Klasse, die niemand umbenannt hat, da die IDE sie nie als Referenz auf `Invoice` erkannt hat. Ein zweites Beispiel: Das Backend einer Single-Page-Application rendert ein Template-Attribut `module="frontend/components/InvoiceTable"`, und eine Bridge löst diesen String in ein dynamisches `import()` im Frontend auf. Als das Frontend-Team seinen Komponentenordner im Zuge eines Aufräumens umstrukturiert, kompiliert TypeScript sauber und die eigenen Tests des Frontends laufen durch, aber der hartcodierte Template-String des Backends zeigt weiterhin auf den alten Pfad, und das Admin-Dashboard rendert in Produktion stillschweigend ein leeres Panel.
